<?php

use Illuminate\Support\Facades\Route;

Route::get('/', fn() => redirect('/dashboard'));

// Панели
Route::get('/dashboard', [\App\Http\Controllers\DashboardController::class, 'index']);
Route::get('/osdr',      [\App\Http\Controllers\OsdrController::class,      'index']);

// Прокси к rust_iss
Route::get('/api/iss/last',  [\App\Http\Controllers\ProxyController::class, 'last']);
Route::get('/api/iss/trend', [\App\Http\Controllers\ProxyController::class, 'trend']);

// JWST галерея (JSON)
Route::get('/api/jwst/feed', [\App\Http\Controllers\DashboardController::class, 'jwstFeed']);
Route::get("/api/astro/events", [\App\Http\Controllers\AstroController::class, "events"]);
use App\Http\Controllers\AstroController;
Route::get('/page/{slug}', [\App\Http\Controllers\CmsController::class, 'page']);
Route::get('/page/{slug}', [\App\Http\Controllers\CmsController::class, 'page']);
