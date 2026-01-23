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
  # Save .env before applying patches to preserve APP_KEY if it exists
  APP_KEY_VALUE=""
  if [ -f "$APP_DIR/.env" ]; then
    APP_KEY_VALUE=$(grep "^APP_KEY=base64:" "$APP_DIR/.env" 2>/dev/null || echo "")
  fi
  # Temporarily remove .env from patches directory if it exists, to prevent overwrite
  PATCH_ENV_BACKUP=""
  if [ -f "$PATCH_DIR/.env" ]; then
    mv "$PATCH_DIR/.env" "$PATCH_DIR/.env.patch" || true
    PATCH_ENV_BACKUP="yes"
  fi
  # Apply patches
  rsync -a "$PATCH_DIR/" "$APP_DIR/" || true
  # Restore .env from patches if we moved it
  if [ -n "$PATCH_ENV_BACKUP" ] && [ -f "$PATCH_DIR/.env.patch" ]; then
    mv "$PATCH_DIR/.env.patch" "$PATCH_DIR/.env" || true
  fi
  # Always ensure .env is valid Laravel .env file
  NEED_RESTORE=0
  if [ ! -f "$APP_DIR/.env" ]; then
    NEED_RESTORE=1
  elif ! grep -q "^APP_KEY=" "$APP_DIR/.env" 2>/dev/null; then
    NEED_RESTORE=1
  fi
  if [ "$NEED_RESTORE" = "1" ]; then
    echo "[php] restoring .env from .env.example"
    if [ -f "$APP_DIR/.env.example" ]; then
      cp "$APP_DIR/.env.example" "$APP_DIR/.env" || true
      sed -i 's|APP_NAME=Laravel|APP_NAME=ISSOSDR|g' "$APP_DIR/.env" || true
      # Restore APP_KEY if we had it before
      if [ -n "$APP_KEY_VALUE" ]; then
        sed -i "s|^APP_KEY=.*|$APP_KEY_VALUE|g" "$APP_DIR/.env" || true
      fi
    fi
  fi
fi

# Ensure APP_KEY is set after patches are applied
if [ -f "$APP_DIR/.env" ]; then
  # Verify .env file is valid (contains APP_KEY= line)
  if ! grep -q "^APP_KEY=" "$APP_DIR/.env" 2>/dev/null; then
    echo "[php] .env file is invalid, restoring from .env.example"
    if [ -f "$APP_DIR/.env.example" ]; then
      cp "$APP_DIR/.env.example" "$APP_DIR/.env" || true
      sed -i 's|APP_NAME=Laravel|APP_NAME=ISSOSDR|g' "$APP_DIR/.env" || true
    fi
  fi
  # Generate key if not set
  if ! grep -q "^APP_KEY=base64:" "$APP_DIR/.env" 2>/dev/null; then
    echo "[php] generating application key"
    php "$APP_DIR/artisan" key:generate --force || true
  fi
fi

chown -R www-data:www-data "$APP_DIR"
chmod -R 775 "$APP_DIR/storage" "$APP_DIR/bootstrap/cache" || true

echo "[php] starting php-fpm"
php-fpm -F
