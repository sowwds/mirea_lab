#!/usr/bin/env bash
set -e

APP_DIR="/var/www/html"
PATCH_DIR="/opt/laravel-patches"

echo "[php] init start"

if [ ! -f "$APP_DIR/artisan" ]; then
  echo "[php] creating laravel skeleton"
  composer create-project --no-interaction --prefer-dist laravel/laravel:^11 "$APP_DIR"
  cp "$APP_DIR/.env.example" "$APP_DIR/.env" || true
  sed -i 's|APP_NAME=Laravel|APP_NAME=ISSOSDR|g' "$APP_DIR/.env" || true
  php "$APP_DIR/artisan" key:generate || true
fi

if [ -d "$PATCH_DIR" ]; then
  echo "[php] applying patches"
  rsync -a "$PATCH_DIR/" "$APP_DIR/"
fi

chown -R www-data:www-data "$APP_DIR"
chmod -R 775 "$APP_DIR/storage" "$APP_DIR/bootstrap/cache" || true

echo "[php] starting php-fpm"
php-fpm -F
