# Контракт training metrics v5

> CONTRACT DOCUMENT. Этот каталог задаёт будущие immutable training records и
> OpenSearch projection consumer-neutral model. Runtime продолжает писать
> metrics v4 до отдельного clean-cut change.

Record v5 хранит D1 data/target/objective/model digests и generic references:
direct loss содержит component identity, operator, opaque target identity и
derived physical index; auxiliary loss содержит component identity/operator;
gradient observations ссылаются на component identities. Identity никогда не
используется как dynamic JSON/OpenSearch field name.

`training-record.schema.json` — durable epoch artifact.
`point.schema.json` — одна конечная числовая OpenSearch observation.
`opensearch/metrics-points-v5.template.json` — future strict mapping с нулём
replicas. Его наличие в repository не устанавливает template и не меняет
deployment.

Full TargetContract/Objective не копируются в каждую observation; их
authoritative copy находится в model/checkpoint metadata, а metrics связывают
её по D1 digests.

`targets`, `directLosses` и `targetMetrics` следуют exact slot order;
`targetIndex` равен позиции, а identity совпадает с checkpoint-owned slot.
Direct component и auxiliary component разрешаются в exact Objective.
Gradient component содержит target identity/index либо оба, либо ни одного;
pair context взаимоисключает одиночные target/component contexts. Все
числовые observations finite. Эти межполевые правила проверяются после JSON
Schema и не зависят от значения opaque identity.
