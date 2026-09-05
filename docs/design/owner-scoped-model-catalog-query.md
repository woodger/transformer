# Owner-scoped Model Catalog Query

> Тип: Design Note. Transformer-side предложение read-only query boundary для
> совместного обсуждения с Consumer-ами. Это не ADR, не wire schema, не
> назначение версии Flight и не описание уже реализованного API.

- Статус: согласовано с Consumer-ом для подготовки wire design
- Срез: 2026-09-05, техническая исходная точка Flight v11
- Входной материал: Consumer-side запрос Inventory и Inventory ADR-0013
- Область изменения: будущий query contract; текущие runtime, contracts,
  schemas, migrations и deployment не изменяются

## Вывод

Потребность обоснована. Источником owner-scoped каталога должен оставаться
Transformer model registry, а не OpenSearch, filesystem scan или Consumer-owned
копия. Registry уже владеет identity и lifecycle опубликованной generation,
точными contracts, lineage, checkpoint identity и связью с producing job.

Минимальная первая композиция состоит из двух концептуальных операций:

1. bounded live high-water traversal возвращает компактные model summaries;
2. single detail остаётся каноническим описанием одной generation;

Inventory ограничивает одно сравнение четырьмя generations, поэтому до четырёх
single-detail запросов являются bounded Consumer behavior. Bounded batch detail
остаётся допустимым последующим ergonomic extension, но не требуется в первой
версии. Все projections читают один registry. List не должен читать OpenSearch
или хэшировать каждый checkpoint; detail применяет строгую проверку model
metadata и checkpoint artifact.

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
- полный resolved `modelConfig`;
- ordered opaque target identities;
- initialization kind и, при наличии, parent `modelRef`;
- producing run identity;
- checkpoint format, SHA-256 и byte count.

Полные `dataContract`/`ModelContract`, training configuration и selection
summary в list не входят. `modelConfig` возвращается как полный validated
provider-owned configuration, а не восстанавливается Consumer-ом из geometry
или других полей.

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

Batch detail не требуется в первой версии. Если он будет добавлен позднее, его
область ограничивается выбранным Consumer-ом набором до четырёх exact
`modelRef`. Он должен:

- иметь advertised maximum items и общий response/work budget;
- запрещать duplicate references;
- сохранять порядок запроса;
- применять к найденной generation ту же validation и physical checkpoint
  verification, что single detail;
- возвращать для каждого отсутствующего элемента одинаковый `not found`
  outcome независимо от того, неизвестен он или принадлежит другому owner;
- не превращаться в unbounded export всего registry.

Server-side comparison не добавляется: правила сопоставления и представления
результатов принадлежат Inventory. Batch только сокращает число transport
round-trips и не вычисляет различия между models.

## Pagination и consistency

### Предпочтительная модель: live high-water + keyset cursor

Первая страница фиксирует верхнюю publication boundary текущего owner-scoped
registry. Все последующие страницы используют opaque cursor, связанный с:

- authenticated owner;
- high-water boundary и canonical ordering;
- последним возвращённым sort key;
- сроком действия cursor.

Согласованный deterministic order:

```text
createdAt DESC, modelRef ASC
```

`modelRef` является safe ASCII identity, поэтому tie-break order не зависит от
locale. Keyset pagination не использует offset и не деградирует линейно с
глубиной каталога.

Гарантии traversal:

- generation, опубликованные после high-water boundary, не появляются в
  текущем traversal;
- одна неизменившаяся generation не появляется дважды;
- publication не меняет порядок уже начатого traversal;
- новый traversal видит актуальный registry state.

Traversal читает только текущие `AVAILABLE` rows. Concurrent deletion может
создать оговорённый пропуск относительно набора, существовавшего на первой
странице; строгая snapshot membership не обещается. Это приемлемая UI
consistency model: Inventory убирает generation из текущего представления,
показывает краткое уведомление и обновляет каталог. Detail и любая runtime
operation после удаления возвращают current owner-scoped `not found`. Старая
telemetry не участвует ни в одном из этих решений.

Cursor является capability с конечным TTL. Expired cursor не продолжает
неполный traversal: Consumer получает structured restart outcome и начинает
новый traversal.

### Непредпочтительный вариант: строгий revisioned snapshot

Строгий snapshot потребовал бы сохранять membership или достаточно полные
revisioned tombstones до expiration cursor. Для интерактивного каталога
Inventory считает эту сложность избыточной. Вариант следует пересматривать
только при появлении export/audit use case с требованием полного repeatable
traversal.

### Непредпочтительный вариант: offset pagination

Offset над изменяемым набором создаёт duplicates и пропуски при concurrent
publish/delete, а стоимость глубоких страниц растёт. Использовать его как
публичную consistency model не предлагается.

## Filters и ordering

Минимальный первый contract не содержит filters или произвольного query
language. Он обходит все owner-visible generations в canonical order.
Дополнительные filters требуют отдельного Consumer use case; при появлении они
должны стать частью cursor identity.

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
- Будущий batch detail использует тот же per-item outcome и не сообщает причину
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

- доступность list и single detail, а при будущем расширении — batch detail;
- query revision;
- consistency model и canonical ordering;
- максимальный page size;
- максимальное число batch details, если operation поддерживается;
- максимальный response/work budget;
- cursor TTL;
- наличие filters и structured per-item outcomes, если они поддерживаются.

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
| Невалидный ordering, page size или batch | Invalid catalog query с path/reason |
| Повреждённый, чужой или подменённый cursor | Invalid cursor |
| Cursor или traversal boundary истекли | Restart-required cursor outcome |
| Query revision или operation не advertised | Capability unavailable |
| Unknown, foreign или уже удалённый `modelRef` | Единый model not found |
| Stored canonical metadata или checkpoint повреждены | Model corrupt |
| Превышен advertised bounded work/response budget | Resource exhausted либо invalid requested limit |
| Registry временно недоступен | Service unavailable |

Человекочитаемый message остаётся диагностикой. Ветвление Consumer выполняет
по stable code/reason и структурным fields. Будущий batch detail должен
отделять ошибку всего запроса от per-item `not found`/`model corrupt`, чтобы
удаление одной selected generation не скрывало результаты остальных.

## Проверка Consumer-side сценариев

1. **Пустой owner catalog.** Первая страница успешно возвращает пустой набор и
   terminal cursor state; это не `NOT_FOUND`.
2. **Несколько labels и generations.** Неизменившиеся rows возвращаются без
   duplicates в canonical order и с bounded page size; concurrent deletion
   может дать оговорённый пропуск.
3. **Одинаковый label.** Generations различаются exact `modelRef` и generation;
   label не используется как persisted selection.
4. **Unknown и foreign reference.** Single и будущий batch detail возвращают
   security-equivalent not found без owner metadata.
5. **Concurrent deletion.** Live traversal допускает оговорённый пропуск;
   последующий detail безопасно возвращает not found. Inventory уведомляет
   пользователя и refresh-ит каталог. Telemetry не сохраняет catalog entry.
6. **Нет OpenSearch documents.** Registry generation остаётся в list/detail и
   пригодна для predict или warm start при успешной integrity validation.
7. **Child lineage.** Доступная child generation возвращает сохранённую parent
   reference; parent detail разрешается только в текущем owner scope.
8. **Повреждённый checkpoint.** List остаётся bounded registry query; detail
   возвращает structured model-corrupt outcome и не выдаёт generation за
   проверенную.
9. **Concurrent publication.** Новая generation выше high-water boundary не
   вмешивается в начатый traversal и появляется после refresh.
10. **Expired cursor.** Consumer получает однозначное указание начать новый
    traversal, а не пустую или частично повторённую страницу.

## Альтернативы

### A. Предпочтительный list + single detail без batch

Минимальный surface и простая реализация. Inventory ограничивает comparison
четырьмя generations, поэтому число integrity-checked detail calls заранее
ограничено на Consumer side.

### B. Optional bounded batch detail

Сокращает до одного round-trip получение максимум четырёх details, но не меняет
семантику и не выполняет server-side comparison. Допустимо как последующее
эргономическое расширение после подтверждения практической пользы.

### C. Rich list без отдельного batch

Каждая строка содержит полный `ModelContract` и training metadata. Устраняет
N+1, но раздувает страницы повторяющимися bounded documents и делает list
дорогим для обычного discovery. Не предпочтительно.

### D. Strict revisioned snapshot

Исключает deletion gaps, но требует temporal membership/tombstones. Inventory
не видит достаточной ценности для исходного Terminal use case.

### E. Catalog из OpenSearch

Отклоняется. Best-effort delivery, независимый retention и сохранение stale
telemetry после удаления противоречат model lifecycle.

### F. Прямой доступ Inventory к PostgreSQL или filesystem

Отклоняется. Раскрывает provider storage schema, owner identities и paths,
обходит application authorization и связывает Consumer с внутренним layout.

### G. Отдельный custom gRPC query service

Технически возможен и получает собственные proto definitions. Пока не
предпочтителен из-за второго public endpoint и дублирования authentication,
errors, capabilities и deployment. Требует повторного рассмотрения, если
catalog станет частью более широкого самостоятельного control plane.

## Зафиксированные позиции Inventory

- List summary имеет согласованный состав, включая полный resolved
  `modelConfig` и ordered target identities, но без полных contracts, training
  configuration и selection state.
- Одновременно сравниваются не более четырёх generations.
- Server-side comparison не требуется; batch detail в первой версии
  необязателен.
- Live high-water traversal достаточен; deletion gap устраняется UI refresh.
- Canonical order — `createdAt DESC, modelRef ASC`.
- Filters и current alias в первой версии не нужны.
- Training configuration и selection summary принадлежат detail.
- `producingRunId` является только correlation identity. Bounded telemetry/run
  query рассматривается как отдельная будущая граница и не блокирует catalog.
- Persisted selection всегда хранит exact immutable `modelRef`.

## Вопросы совместного решения

- точная canonical форма high-water cursor и его expiration outcome;
- точная canonical форма согласованной summary projection;
- work budget physical checkpoint verification;
- query capability/version lifecycle относительно Flight workflow;
- срок жизни cursor и advertised page limit;
- необходимость batch detail после проверки первой интеграции.

Consumer-side boundary достаточна для следующего этапа. Теперь Transformer
может подготовить canonical wire proposal: выбрать exact cursor document,
page limits, structured errors, capability lifecycle и место query относительно
Flight workflow, а затем передать schemas и cross-project fixtures Inventory на
review. Код, PostgreSQL migration и deployment следует менять только после
принятия этого package.
