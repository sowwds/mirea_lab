#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
temp_root=$(mktemp -d /tmp/prac1-hal.XXXXXX)
temp_sketch="$temp_root/prac1"
output_dir="$project_dir/.build"

cleanup() {
  rm -rf "$temp_root"
}
trap cleanup EXIT

mkdir "$temp_sketch"
cp "$project_dir/main.c" "$temp_sketch/main.c"
# Arduino CLI требует .ino с именем папки, но он не является исходником проекта.
touch "$temp_sketch/prac1.ino"
mkdir -p "$output_dir"
rm -f "$output_dir"/*.ino.bin "$output_dir"/*.ino.elf "$output_dir"/*.ino.hex "$output_dir"/*.ino.map

arduino-cli compile \
  --fqbn STMicroelectronics:stm32:Nucleo_64:pnum=NUCLEO_C031C6 \
  --output-dir "$output_dir" \
  "$temp_sketch"
