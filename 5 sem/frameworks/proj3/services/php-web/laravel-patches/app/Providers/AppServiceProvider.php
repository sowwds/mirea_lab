<?php

namespace App\Providers;

use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\RateLimiter;
use Illuminate\Cache\RateLimiting\Limit;
use Illuminate\Http\Request;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        //
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        // Настройка rate limiting для API с использованием Redis
        RateLimiter::for('api', function (Request $request) {
            return Limit::perMinute(60)->by($request->user()?->id ?: $request->ip());
        });

        // Строгий лимит для внешних API прокси
        RateLimiter::for('external-api', function (Request $request) {
            return Limit::perMinute(10)->by($request->ip());
        });

        // Лимит для загрузки файлов
        RateLimiter::for('uploads', function (Request $request) {
            return Limit::perMinute(5)->by($request->ip());
        });
    }
}

