#!/bin/sh
# One snapshot while the source is publishing. STDOUT is suitable for tee.
set -eu
ids=$(docker compose --profile observability ps -q loki vector tempo collector metrics-collector prometheus grafana)
if [ "$(echo "$ids" | awk 'NF {n++} END {print n+0}')" != 7 ]; then
    echo "Запустите полный профиль и дождитесь семи сервисов наблюдаемости." >&2
    exit 1
fi
echo "Снимок под нагрузкой, UTC $(date -u '+%F %T')"
# Docker container IDs have no whitespace.
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' $ids | awk '
function mib(s,    n) {
    n = s + 0
    if (s ~ /GiB$/) return n * 1024
    if (s ~ /MiB$/) return n
    if (s ~ /KiB$/) return n / 1024
    return n / 1048576
}
{ print; total += mib($2); count++ }
END {
    if (count != 7) exit 1
    printf "Суммарное потребление %.1f MiB из 1472 MiB лимитов\n", total
}'
