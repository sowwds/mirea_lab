<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

return new class extends Migration {
    public function up(): void
    {
        if (!Schema::hasTable('cms_blocks')) {
            Schema::create('cms_blocks', function (Blueprint $table) {
                $table->id();
                $table->string('slug')->unique();
                $table->string('title');
                $table->text('content');
                $table->boolean('is_active')->default(true);
                $table->timestamps();
            });

            // Seed initial data
            DB::table('cms_blocks')->insert([
                [
                    'slug' => 'demo',
                    'title' => 'Демо страница',
                    'content' => '<h1>Демо контент CMS</h1><p>Это демонстрационная страница из таблицы cms_blocks</p>',
                    'is_active' => true,
                    'created_at' => now(),
                    'updated_at' => now(),
                ],
                [
                    'slug' => 'welcome',
                    'title' => 'Добро пожаловать',
                    'content' => '<h3>Демо контент</h3><p>Этот текст хранится в БД</p>',
                    'is_active' => true,
                    'created_at' => now(),
                    'updated_at' => now(),
                ],
                [
                    'slug' => 'dashboard_experiment',
                    'title' => 'Эксперимент',
                    'content' => '<p>Экспериментальный контент для dashboard</p>',
                    'is_active' => true,
                    'created_at' => now(),
                    'updated_at' => now(),
                ],
            ]);
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('cms_blocks');
    }
};

