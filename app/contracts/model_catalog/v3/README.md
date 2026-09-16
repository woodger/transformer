# Запрос каталога моделей в области владельца v3

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет discovery только для чтения и
> detail опубликованных generations моделей Transformer. Он активируется
> Flight v15.

Registry — источник истины catalog. Telemetry не является ни источником
существования, ни условием видимости модели. Requests никогда не передают
owner-а: Transformer выводит scope из аутентифицированного subject. Неизвестные,
чужие и удалённые `modelRef` security-equivalent и возвращают
`MODEL_NOT_FOUND`.

## Операции

```text
transformer.model-catalog.v3.list
transformer.model-catalog.v3.detail
```

List — ограниченный owner-scoped traversal, упорядоченный по `createdAt DESC,
modelRef ASC`. Он использует подписанный, связанный с owner-ом keyset cursor с
TTL 900 секунд и границей live high-water. Параллельное удаление может привести
к `MODEL_NOT_FOUND` при позднейшем lookup detail; Consumer должен убрать эту
запись и обновить данные. Cursor не сохраняет доступ к удалённой generation.

Каждый summary list содержит только данные, требуемые для выбора и сравнения
generation: immutable `modelRef`, label, generation, время создания,
непрозрачные identities data/model definition, разрешённый model tuning,
упорядоченные непрозрачные target identities, summary initialization catalog и
identity producing run. Он не раскрывает bytes/hashes checkpoint-а, filesystem
paths, версии Worker или полные документы target/objective.

Detail принимает один точный `modelRef`. Он возвращает summary и:

- digest data и geometry tensor-а;
- полный semantic v3 `ModelContract` и все слои D1;
- разрешённые settings training/diagnostics;
- terminal progress и summary selection; и
- summary initialization catalog (`random` либо `publishedModel` с видимой
  reference parent model).

Запрошенная initialization, разрешённые checkpoint-ом hashes parent-а, paths
checkpoint-а и byte counts artifact остаются внутренними для provider-а.
Ответ detail — каноническое описание generation, а не API администрирования
artifacts.

## Проверка и ошибки

List читает только metadata registry. Detail валидирует сохранённые metadata и
D1 до проверки managed artifact checkpoint-а. Некорректная сохранённая
definition возвращает `MODEL_CORRUPT / STORED_MODEL_METADATA_INVALID`;
некорректный artifact — `MODEL_CORRUPT / MODEL_CHECKPOINT_INVALID`; limit
бюджета работы verification возвращает
`RESOURCE_EXHAUSTED / MODEL_VERIFICATION_UNAVAILABLE`.

Другие structured outcomes включают некорректный/истёкший cursor,
некорректный request, недоступность registry и исчерпание response budget.
Документы ошибок определены в `schemas/error-detail.schema.json`; clients
ветвятся по `code` и `reason`, но никогда по читаемому человеку тексту.

Максимальный размер страницы — 100, TTL cursor — 900 секунд, а сериализованный
result ограничен 8 MiB. Detail проверяет не более одного checkpoint-а и
отказывается от verification более 1 GiB до его чтения.

## Чистый переход и fixtures

Revision 3 не имеет reader для entries catalog v1/v2. Migration 0027 Flight
v15 удаляет предыдущие generations и их records registry, поэтому deployment
начинает с пустого catalog, а новые models обучаются в semantic v3.

`fixtures/` содержит примеры empty/list/detail для офлайн межпроектной
проверки. Его manifest хеширует только этот bundle fixtures и не имеет
значения для runtime или compatibility.
