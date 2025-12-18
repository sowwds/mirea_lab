<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        Schema::table('telemetry_legacy', function (Blueprint $table) {
            if (!Schema::hasColumn('telemetry_legacy', 'flag_a')) {
                $table->boolean('flag_a')->default(false)->after('recorded_at');
            }
            if (!Schema::hasColumn('telemetry_legacy', 'flag_b')) {
                $table->boolean('flag_b')->default(false)->after('flag_a');
            }
            if (!Schema::hasColumn('telemetry_legacy', 'count')) {
                $table->integer('count')->default(0)->after('temp');
            }
            if (!Schema::hasColumn('telemetry_legacy', 'note')) {
                $table->text('note')->nullable()->after('count');
            }
        });
    }

    public function down(): void
    {
        Schema::table('telemetry_legacy', function (Blueprint $table) {
            if (Schema::hasColumn('telemetry_legacy', 'note')) {
                $table->dropColumn('note');
            }
            if (Schema::hasColumn('telemetry_legacy', 'count')) {
                $table->dropColumn('count');
            }
            if (Schema::hasColumn('telemetry_legacy', 'flag_b')) {
                $table->dropColumn('flag_b');
            }
            if (Schema::hasColumn('telemetry_legacy', 'flag_a')) {
                $table->dropColumn('flag_a');
            }
        });
    }
};
