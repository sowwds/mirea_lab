<?php

namespace App\Http\Controllers;

use App\Http\Requests\TelemetryListRequest;
use App\Models\TelemetryLegacy;

class TelemetryController extends Controller
{
    public function index()
    {
        return view('telemetry.index');
    }

    public function list(TelemetryListRequest $request)
    {

        $q = TelemetryLegacy::query();

        if ($request->filled('search')) {
            $search = $request->input('search');
            $q->where(function($sub) use ($search) {
                $sub->where('note', 'ilike', "%{$search}%")
                    ->orWhere('source_file', 'ilike', "%{$search}%");
            });
        }

        if ($request->filled('keywords')) {
            $words = array_filter(preg_split('~\s+~u', $request->input('keywords')), fn($w) => mb_strlen($w) > 1);
            if ($words) {
                $q->where(function($sub) use ($words) {
                    foreach ($words as $w) {
                        $sub->where(function($x) use ($w) {
                            $x->where('note', 'ilike', "%{$w}%")
                              ->orWhere('source_file', 'ilike', "%{$w}%");
                        });
                    }
                });
            }
        }

        if ($request->filled('from')) {
            $q->where('recorded_at', '>=', $request->input('from'));
        }
        if ($request->filled('to')) {
            $q->where('recorded_at', '<=', $request->input('to'));
        }

        if ($request->filled('flag_a')) {
            $q->where('flag_a', filter_var($request->input('flag_a'), FILTER_VALIDATE_BOOLEAN));
        }
        if ($request->filled('flag_b')) {
            $q->where('flag_b', filter_var($request->input('flag_b'), FILTER_VALIDATE_BOOLEAN));
        }

        $allowedSorts = ['recorded_at','voltage','temp','count'];
        $sort = $request->input('sort', 'recorded_at');
        $dir = strtolower($request->input('dir', 'desc')) === 'asc' ? 'asc' : 'desc';
        if (!in_array($sort, $allowedSorts)) $sort = 'recorded_at';
        $q->orderBy($sort, $dir);

        $perPage = (int) $request->input('per_page', 25);

        $result = $q->paginate($perPage)->withQueryString();

        return response()->json($result);
    }
}
