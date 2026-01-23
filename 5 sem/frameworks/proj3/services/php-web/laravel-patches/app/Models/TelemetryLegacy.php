<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class TelemetryLegacy extends Model
{
    protected $table = 'telemetry_legacy';

    protected $fillable = [
        'recorded_at','flag_a','flag_b','voltage','temp','count','note','source_file'
    ];

    protected $casts = [
        'recorded_at' => 'datetime',
        'flag_a' => 'boolean',
        'flag_b' => 'boolean',
        'voltage' => 'decimal:2',
        'temp' => 'decimal:2',
        'count' => 'integer',
    ];
}
