# Owner-scoped Model Catalog Query

> Тип: Design Note. Transformer-side предложение read-only query boundary для
> совместного обсуждения с Consumer-ами. Это не ADR, не wire schema, не
> назначение версии Flight и не описание уже реализованного API.

- Статус: proposal к cross-project review
- Срез: 2026-09-05, техническая исходная точка Flight v11
- Входной материал: Consumer-side запрос Inventory и Inventory ADR-0013
- Область изменения: будущий query contract; текущие runtime, contracts,
  schemas, migrations и deployment не изменяются

## Вывод

Потребность обоснована. Источником owner-scoped каталога должен оставаться
Transformer model registry, а не OpenSearch, filesystem scan или Consumer-owned
копия. Registry уже владеет identity и lifecycle опубликованной generation,
точными contracts, lineage, checkpoint identity и связью с producing job.

Предпочтительная композиция состоит из трёх концептуальных операций:

1. bounded snapshot traversal возвращает компактные model summaries;
2. single detail остаётся каноническим описанием одной generation;
3. bounded batch detail позволяет сравнить несколько выбранных `modelRef` без
   неограниченного N+1.

Это разные projections одного registry, а не три источника истины. List не
должен читать OpenSearch или хэшировать каждый checkpoint. Single и batch
detail должны применять одинаковую строгую проверку model metadata и
checkpoint artifact.

Arrow data plane, `indexedFeatureBlocks`, fit, predict, objective language и
model lifecycle этим предложением не меняются. По характеру операция подходит
для read-only control plane. Предпочтительным transport-кандидатом остаётся
Flight `DoAction`, но wire surface и его версия определяются только после
cross-project согласования.

## Проверенное текущее состояние

PostgreSQL registry уже хранит для каждой published generation:

- `modelRef`, authenticated owner subject, label и generation;
- lifecycle state и creation time;
- checkpoint path, byte count и SHA-256;
- canonical `dataContract` и `ModelContract`;
- D1 semantic digests;
- checkpoint-owned metadata;
- producing job identity.

Текущий `model.describe`:

- разрешает `modelRef` или текущий alias только внутри authenticated owner;
- рассматривает unknown и foreign generation одинаково как not found;
- возвращает только generation в состоянии `AVAILABLE`;
- повторно валидирует canonical contracts и D1 digests;
- проверяет наличие, размер и SHA-256 checkpoint artifact;
- возвращает contracts, lineage и checkpoint summary, но пока не возвращает
  resolved training configuration, selection summary или producing run
  identity.

Административный `models list` не является основой публичного query: он
показывает модели всех owners, не имеет pagination и предназначен для локальной
операторской команды. Его проекция и authorization boundary отличаются от
потребности Consumer-а.

Training telemetry публикуется независимо и best effort. OpenSearch documents
коррелируют run по `runId`, равному producing job identity. Отсутствие или
задержка этих documents не меняют registry state. После удаления generation
telemetry может сохраниться, но не должна создавать catalog entry.

## Источник истины и ownership

| Понятие | Источник истины | Query outcome |
| --- | --- | --- |
| Существование доступной generation | Transformer model registry | Catalog list и detail |
| `modelRef`, label, generation, creation time | Registry row | Summary и detail |
| Exact data/model semantics | Canonical documents registry/checkpoint | Detail; digests и bounded summary в list |
| Resolved training configuration | Checkpoint-owned immutable metadata | Detail |
| Final selection state и training progress | Metadata опубликованного checkpoint | Detail summary |
| Checkpoint format, digest и bytes | Registry с проверкой artifact на detail | Summary и detail |
| Initialization lineage | Immutable model metadata | Summary и detail |
| Producing run identity | Registry relation к producing job | Summary и detail |
| Epoch metrics, timings и input counters | Run telemetry | Не являются catalog truth |
| UI grouping, descriptions и comparison report | Inventory | Не materialize-ятся Transformer-ом |

`dataContract`, `ModelContract`, training configuration и selection state
принадлежат конкретной immutable model generation, даже если их исходный выбор
частично пришёл от Consumer-а. Они определяют исполняемый checkpoint и поэтому
могут быть возвращены без обращения к telemetry.

Lifecycle durations, attempts, recoveries, input payload counters и epoch
observations описывают run, а не саму generation. Catalog возвращает стабильный
producing run identity; присоединение доступной telemetry выполняется по этой
identity и не влияет на наличие модели.

## Концептуальные projections

Имена ниже обозначают семантические сущности, а не будущие JSON fields.

### Model summary

Одна list entry должна быть достаточно полной для каталога и предварительного
сравнения, но не копировать целые canonical documents:

- exact `modelRef`;
- display label и generation number;
- creation time;
- четыре D1 semantic digests;
- geometry summary: `seqLen`, `featureDim` и число target slots;
- ordered opaque target identities;
- initialization kind и, при наличии, parent `modelRef`;
- producing run identity;
- checkpoint format, SHA-256 и byte count.

Summary не является доказательством физической целостности checkpoint в момент
чтения. Он отражает committed registry state. Хэширование всех checkpoint-ов
на каждой странице сделало бы стоимость list пропорциональной общему объёму
weights и превратило discovery в неограниченную filesystem operation.

### Model detail

Detail является полным checkpoint-owned описанием и должен включать:

- все поля summary;
- canonical `dataContract` и `ModelContract`;
- resolved training configuration и diagnostics configuration;
- final selection summary: включённость, выбранная epoch/score и источник
  published weights;
- terminal training progress: completed epochs и global step;
- exact initialization lineage;
- job configuration digest и producing run identity;
- checkpoint summary после проверки physical artifact.

В detail не входят:

- owner subject, поскольку scope уже задан authentication;
- checkpoint path и filesystem topology;
- optimizer, AMP scaler, RNG или tensor state;
- значения private resources;
- metric points, run durations и OpenSearch availability;
- Consumer-owned descriptions, formulas или UI labels кроме сохранённой opaque
  identity.

Текущий `model.describe` является естественной семантической основой detail.
Нужно расширять его outcome, а не создавать второе несовместимое понятие полной
model metadata.

### Bounded batch detail

Batch detail нужен только для выбранного Consumer-ом небольшого набора exact
`modelRef`. Он должен:

- иметь advertised maximum items и общий response/work budget;
- запрещать duplicate references;
- сохранять порядок запроса;
- применять к найденной generation ту же validation и physical checkpoint
  verification, что single detail;
- возвращать для каждого отсутствующего элемента одинаковый `not found`
  outcome независимо от того, неизвестен он или принадлежит другому owner;
- не превращаться в unbounded export всего registry.

Если фактические UI-сценарии укладываются в один detail за раз, batch operation
можно отложить. Наличие list и single detail не должно, однако, закреплять
неограниченный клиентский N+1 как единственный способ сравнения.

## Pagination и consistency

### Предпочтительная модель: revisioned snapshot + keyset cursor

Первая страница фиксирует owner-scoped registry snapshot revision. Все
последующие страницы используют opaque cursor, связанный с:

- authenticated owner;
- snapshot revision;
- normalized filter и ordering;
- последним возвращённым sort key;
- сроком действия cursor.

Предпочтительный deterministic order для исходного use case:

```text
label ASC, generation DESC, modelRef ASC
```

Label и `modelRef` являются safe ASCII identities, поэтому порядок можно
определить независимо от locale. Keyset pagination не использует offset и не
деградирует линейно с глубиной каталога.

Гарантии snapshot:

- generation, опубликованные после snapshot revision, не появляются в
  текущем traversal;
- одна snapshot member не появляется дважды;
- publication или deletion не меняют порядок уже выбранного snapshot;
- новый traversal видит актуальный registry state.

Для строгой гарантии membership registry должен сохранять достаточный
revisioned list projection как минимум до expiration cursor. Это может быть
temporal registry state, revisioned tombstone или bounded materialized
snapshot. Design Note не выбирает persistence representation.

Удаление после list, но до detail имеет безопасную семантику: snapshot list
может содержать generation, существовавшую на момент snapshot, а detail и
любая runtime operation возвращают current owner-scoped `not found`. Fresh
traversal удалённую generation не показывает. Старая telemetry не участвует ни
в одном из этих решений.

Cursor является capability с конечным TTL. Expired cursor не продолжает
неполный traversal: Consumer получает structured restart outcome и начинает
новый snapshot.

### Более простой допустимый вариант: live high-water keyset

Первый page фиксирует верхнюю publication boundary, а дальнейшие страницы
читают только текущие `AVAILABLE` rows ниже неё. Этот вариант исключает новые
publication и duplicates, но deletion во время traversal может создать
оговорённый пропуск относительно первой страницы. Он не требует хранения
snapshot membership и может быть достаточен, если Inventory явно принимает
refresh-oriented UI consistency.

### Непредпочтительный вариант: offset pagination

Offset над изменяемым набором создаёт duplicates и пропуски при concurrent
publish/delete, а стоимость глубоких страниц растёт. Использовать его как
публичную consistency model не предлагается.

## Filters и ordering

Минимальный первый contract не нуждается в произвольном query language.
Достаточно traversal всех owner-visible generations и, если это подтверждает
Inventory, exact label filter. Filter становится частью cursor identity.

Server-side sorting по loss, метрикам или Consumer descriptions запрещён:
такие значения не принадлежат registry и могут отсутствовать. Inventory может
сортировать выбранный bounded comparison set после присоединения telemetry.

## Security model

- Owner scope выводится только из authenticated subject.
- Request не принимает owner identity, tenant ID или bypass flag.
- Cursor должен быть opaque, integrity-protected и связан с owner subject;
  повторное использование другим owner отклоняется без раскрытия contents.
- List никогда не возвращает foreign generation.
- Unknown, deleted и foreign `modelRef` имеют security-equivalent detail
  outcome.
- Batch detail использует тот же per-item outcome и не сообщает причину
  отсутствия.
- Label не является selector-ом persisted выбора: Consumer сохраняет только
  exact `modelRef`.
- Parent lineage возвращается как часть metadata доступной child generation.
  Dereference parent выполняется отдельно через тот же owner-scoped detail;
  удалённый parent не восстанавливается из lineage.
- Checkpoint paths, credentials, private tensor state, database keys и
  OpenSearch topology не пересекают boundary.

## Capability и limits

Model Catalog Query должен иметь собственную language/query revision,
независимую от objective language revision. Capabilities должны позволять
Consumer-у определить:

- доступность list, single detail и batch detail;
- query revision;
- consistency model и canonical ordering;
- максимальный page size;
- максимальное число batch details;
- максимальный response/work budget;
- cursor TTL и поддержку filters;
- наличие structured per-item outcomes.

Добавление нового target, operator или Arrow encoding не меняет query revision.
Изменение cursor semantics, list membership, ordering или значения уже
существующего detail field требует новой query revision.

Flight workflow не обязан семантически версионироваться вместе с каждым
изменением catalog query. Однако текущий Flight v11 имеет закрытые schemas и
замороженный action/capabilities surface, поэтому новый action нельзя молча
добавить в нормативный v11 package. На wire-design этапе нужно выбрать один из
вариантов:

1. включить независимо versioned catalog query в следующий Flight contract;
2. определить общий extension lifecycle, если он будет нужен не только
   catalog query;
3. вынести более широкий read-only control plane в собственный gRPC service.

Для одной catalog capability третий вариант непропорционален: он добавит
отдельные proto, listener, TLS/auth, deployment и client runtime. Он становится
обоснованным только если формируется самостоятельный набор provider-owned
control-plane queries. Предварительно предпочтителен Flight `DoAction` с
отдельной query revision.

## Structured outcomes

Точная wire taxonomy ещё не определяется. Семантически Consumer должен уметь
различить без parsing message:

| Ситуация | Категория outcome |
| --- | --- |
| Невалидный filter, ordering, page size или batch | Invalid catalog query с path/reason |
| Повреждённый, чужой или подменённый cursor | Invalid cursor |
| Cursor или snapshot истёк | Restart-required cursor outcome |
| Query revision или operation не advertised | Capability unavailable |
| Unknown, foreign или уже удалённый `modelRef` | Единый model not found |
| Stored canonical metadata или checkpoint повреждены | Model corrupt |
| Превышен advertised bounded work/response budget | Resource exhausted либо invalid requested limit |
| Registry временно недоступен | Service unavailable |

Человекочитаемый message остаётся диагностикой. Ветвление Consumer выполняет
по stable code/reason и структурным fields. Batch detail должен отделять
ошибку всего запроса от per-item `not found`/`model corrupt`, чтобы удаление
одной selected generation не скрывало результаты остальных.

## Проверка Consumer-side сценариев

1. **Пустой owner catalog.** Первая страница успешно возвращает пустой набор,
   terminal cursor state и snapshot identity; это не `NOT_FOUND`.
2. **Несколько labels и generations.** Все snapshot members возвращаются ровно
   один раз в canonical order и с bounded page size.
3. **Одинаковый label.** Generations различаются exact `modelRef` и generation;
   label не используется как persisted selection.
4. **Unknown и foreign reference.** Single и batch detail возвращают
   security-equivalent not found без owner metadata.
5. **Concurrent deletion.** Snapshot traversal следует объявленной consistency
   model; последующий detail безопасно возвращает not found. Telemetry не
   сохраняет catalog entry.
6. **Нет OpenSearch documents.** Registry generation остаётся в list/detail и
   пригодна для predict или warm start при успешной integrity validation.
7. **Child lineage.** Доступная child generation возвращает сохранённую parent
   reference; parent detail разрешается только в текущем owner scope.
8. **Повреждённый checkpoint.** List остаётся bounded registry query; detail
   возвращает structured model-corrupt outcome и не выдаёт generation за
   проверенную.
9. **Concurrent publication.** Новая generation не вмешивается в начатый
   snapshot и появляется после refresh.
10. **Expired cursor.** Consumer получает однозначное указание начать новый
    traversal, а не пустую или частично повторённую страницу.

## Альтернативы

### A. List + single detail без batch

Минимальный surface и простая реализация. Подходит, если карточка открывается
редко и сравниваются одна-две модели. Риск — UI закрепит N+1 calls и
параллельное неограниченное хэширование checkpoint-ов.

### B. Rich list без отдельного batch

Каждая строка содержит полный `ModelContract` и training metadata. Устраняет
N+1, но раздувает страницы повторяющимися bounded documents и делает list
дорогим для обычного discovery. Не предпочтительно.

### C. Предпочтительный list + single detail + bounded batch detail

Разделяет дешёвый discovery и дорогую integrity-checked detail operation,
сохраняет bounded comparison и один источник semantics.

### D. Catalog из OpenSearch

Отклоняется. Best-effort delivery, независимый retention и сохранение stale
telemetry после удаления противоречат model lifecycle.

### E. Прямой доступ Inventory к PostgreSQL или filesystem

Отклоняется. Раскрывает provider storage schema, owner identities и paths,
обходит application authorization и связывает Consumer с внутренним layout.

### F. Отдельный custom gRPC query service

Технически возможен и получает собственные proto definitions. Пока не
предпочтителен из-за второго public endpoint и дублирования authentication,
errors, capabilities и deployment. Требует повторного рассмотрения, если
catalog станет частью более широкого самостоятельного control plane.

## Обратные вопросы Inventory

До wire design Transformer требуется позиция Inventory по следующим вопросам:

1. Какие summary fields реально нужны в таблице до открытия карточки?
2. Нужно ли показывать ordered target identities уже в list или достаточно
   target count и target digest?
3. Какое максимальное число generations пользователь сравнивает одновременно?
4. Достаточен ли batch detail для comparison или Inventory ожидает отдельную
   server-side comparison operation?
5. Приемлема ли строгая snapshot consistency с cursor TTL и обязательным
   restart после expiration, либо достаточно live high-water traversal?
6. Нужен ли exact label filter в первой версии? Другие filters должны быть
   обоснованы конкретным UI-сценарием.
7. Должны ли training configuration и selection summary отображаться только в
   detail либо также в list comparison projection?
8. Достаточен ли `producingRunId` для присоединения telemetry через
   Inventory-owned integration, или требуется отдельный provider query
   доступной run summary?
9. Как UI должен показывать generation, удалённую между list и detail: убрать
   после refresh или оставить краткое transient уведомление?
10. Нужно ли отображать current alias для label, если persisted selection всё
    равно всегда использует exact `modelRef`?

## Вопросы совместного решения

- strict snapshot или live high-water consistency;
- exact canonical ordering и минимальный filter set;
- состав summary projection;
- необходимость batch detail с первой версии;
- work budget physical checkpoint verification;
- query capability/version lifecycle относительно Flight workflow;
- способ получения optional run summary без превращения telemetry в registry;
- срок жизни cursor и ожидаемое поведение Consumer при expiration.

После согласования этих вопросов можно подготовить canonical query proposal и
cross-project fixtures. До этого не следует назначать action names, JSON
fields, Flight version, PostgreSQL migration или OpenSearch изменения.
