# ADR 0019: единый источник operational defaults

- Статус: принято
- Дата: 2026-08-23
- Частично пересматривает: ADR 0008

## Контекст

ADR 0008 разместил configuration modules у runtime-владельцев и удалил общий
`app/config.py`. Это сохранило ясные границы загрузки и валидации, но
одновременно распределило встроенные operational defaults между local,
service, PostgreSQL и OpenSearch packages. Одинаковые local/worker defaults
для device и framed Arrow limit также пришлось определять более одного раза.

Для изменения штатного поведения разработчику снова требуется искать значения
по нескольким adapters и bootstrap modules. При этом сами configuration types,
environment loaders и проверки действительно зависят от технологии и должны
оставаться у runtime-владельцев.

Worker model/training configuration имеет другую роль: это версионируемый
межпроцессный и persisted contract, а не deployment configuration.

## Решение

`app/config.py` является единственным источником встроенных operational
defaults, не принадлежащих версионируемому contract. В нём находятся значения
для local runtime, Flight service, PostgreSQL connection defaults и OpenSearch
metrics transport. Модуль содержит только immutable constants, стандартную
библиотеку и зависимость от project identity; он не загружает environment,
credentials или adapters и не создаёт mutable configuration state.

Runtime-владельцы сохраняют configuration types, parsing, validation и
технологические преобразования:

- `app/service/bootstrap/config.py` строит и проверяет Flight service config;
- PostgreSQL adapter загружает `.env`/process environment и создаёт SQLAlchemy
  URL;
- OpenSearch adapter проверяет endpoint, credentials и TLS profile;
- local и worker code используют общие device и framed Arrow defaults.

Приоритеты источников не меняются. Flight service продолжает читать только
поддерживаемые `TRANSFORMER_*` process environment variables и CLI overrides.
Для PostgreSQL и OpenSearch process environment имеет приоритет над project
`.env`. Credentials остаются только в environment sources и runtime objects.

`app/project.py` продолжает владеть project identity и root path.
`app/contracts/worker/v7/config.py` продолжает владеть model/training defaults,
валидацией и serialization. Default `max_payloads_per_job` по-прежнему
определяется нормативным Flight v5 contract. Эти contract values не
дублируются в `app/config.py`.

`app/local/config.py` удаляется без compatibility façade. Остальные
runtime-specific modules с `config` в имени не являются альтернативными
источниками defaults: они применяют и проверяют значения центрального модуля.

## Последствия

- все изменяемые operational defaults видны в одном Python-файле;
- local, service, admin и worker processes загружают только нужные им runtime
  modules;
- adapters не протаскиваются через общий import path;
- изменение default требует проверки всех runtime-потребителей этого значения;
- часть решения ADR 0008 о полном отсутствии общего `app/config.py` больше не
  применяется, остальные runtime ownership boundaries сохраняются.
