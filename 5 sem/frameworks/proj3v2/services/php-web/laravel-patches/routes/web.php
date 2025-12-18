<?php

use App\Http\Controllers\AstroController;
use App\Http\Controllers\CmsController;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\IssController;
use App\Http\Controllers\OsdrController;
use App\Http\Controllers\ProxyController;
use App\Http\Controllers\TelemetryController;
use App\Http\Controllers\UploadController;

Route::get('/', fn () => redirect()->route('dashboard'));

// Панели (базовый rate limit: 60 запросов в минуту)
Route::middleware(['throttle:60,1'])->group(function () {
    Route::get('/dashboard', [DashboardController::class, 'index'])->name('dashboard');
    Route::get('/osdr', [OsdrController::class, 'index'])->name('osdr.index');
    Route::get('/iss', [IssController::class, 'index'])->name('iss.index');
    Route::get('/cms/page/{slug}', [CmsController::class, 'show'])->name('cms.page');
    Route::get('/telemetry', [TelemetryController::class, 'index'])->name('telemetry.index');
});

// API endpoints с более строгим rate limit
Route::middleware(['throttle:api'])->group(function () {
    // Прокси к rust_iss (30 запросов в минуту)
    Route::middleware(['throttle:30,1'])->group(function () {
        Route::get('/api/iss/last', [ProxyController::class, 'last']);
        Route::get('/api/iss/trend', [ProxyController::class, 'trend']);
    });

    // JWST и AstronomyAPI (20 запросов в минуту)
    Route::middleware(['throttle:20,1'])->group(function () {
        Route::get('/api/jwst/feed', [DashboardController::class, 'jwstFeed']);
        Route::get('/api/astro/events', [AstroController::class, 'events']);
    });

    // Телеметрия API (40 запросов в минуту)
    Route::middleware(['throttle:40,1'])->group(function () {
        Route::get('/api/telemetry', [TelemetryController::class, 'list'])->name('api.telemetry.list');
        Route::get('/dashboard/data', [DashboardController::class, 'data'])->name('dashboard.data');
    });
});

// Загрузка файлов (10 запросов в минуту)
Route::middleware(['throttle:10,1'])->group(function () {
    Route::post('/upload', [UploadController::class, 'store'])->name('upload.store');
});