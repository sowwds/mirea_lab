<?php

namespace App\Http\Controllers;

use App\Support\JwstHelper;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class DashboardController extends Controller
{
    private function base(): string
    {
        return getenv('RUST_BASE') ?: 'http://rust_iss:3000';
    }

    private function getJson(string $url, array $qs = []): array
    {
        if ($qs) {
            $url .= (str_contains($url, '?') ? '&' : '?') . http_build_query($qs);
        }
        $raw = @file_get_contents($url);
        return $raw ? (json_decode($raw, true) ?: []) : [];
    }

    public function index()
    {
        // Берём актуальные данные МКС из rust_iss и отдаём в готовый дашборд
        $b   = $this->base();
        $iss = $this->getJson($b . '/last');

        return view('dashboard', [
            'iss'  => $iss,
            'trend' => [],
            'jw_gallery' => [],
            'jw_observation_raw' => [],
            'jw_observation_summary' => [],
            'jw_observation_images' => [],
            'jw_observation_files' => [],
            'metrics' => [
                'iss_speed' => $iss['payload']['velocity'] ?? null,
                'iss_alt'   => $iss['payload']['altitude'] ?? null,
                'neo_total' => 0,
            ],
        ]);
    }

    /**
     * /api/jwst/feed — серверный прокси/нормализатор JWST картинок.
     * QS:
     *  - source: jpg|suffix|program (default jpg)
     *  - suffix: напр. _cal, _thumb, _crf
     *  - program: ID программы (число)
     *  - instrument: NIRCam|MIRI|NIRISS|NIRSpec|FGS
     *  - page, perPage
     */
    public function jwstFeed(\App\Http\Requests\JwstFeedRequest $r)
    {

        $src   = $r->query('source', 'jpg');
        $sfx   = trim((string) $r->query('suffix', ''));
        $prog  = trim((string) $r->query('program', ''));
        $instF = strtoupper(trim((string) $r->query('instrument', '')));
        $page  = max(1, (int) $r->query('page', 1));
        $per   = max(1, min(60, (int) $r->query('perPage', 24)));

        $jw = new JwstHelper();

        // выбираем эндпоинт
        $path = 'all/type/jpg';
        if ($src === 'suffix' && $sfx !== '') $path = 'all/suffix/' . ltrim($sfx, '/');
        if ($src === 'program' && $prog !== '') $path = 'program/id/' . rawurlencode($prog);

        $resp = $jw->get($path, ['page' => $page, 'perPage' => $per]);
        $list = $resp['body'] ?? ($resp['data'] ?? (is_array($resp) ? $resp : []));

        $items = [];
        foreach ($list as $it) {
            if (!is_array($it)) continue;

            // выбираем валидную картинку
            $url = null;
            $loc = $it['location'] ?? $it['url'] ?? null;
            $thumb = $it['thumbnail'] ?? null;
            foreach ([$loc, $thumb] as $u) {
                if (is_string($u) && preg_match('~\.(jpg|jpeg|png)(\?.*)?$~i', $u)) {
                    $url = $u;
                    break;
                }
            }
            if (!$url) {
                $url = \App\Support\JwstHelper::pickImageUrl($it);
            }
            if (!$url) continue;

            // фильтр по инструменту
            $instList = [];
            foreach (($it['details']['instruments'] ?? []) as $I) {
                if (is_array($I) && !empty($I['instrument'])) $instList[] = strtoupper($I['instrument']);
            }
            if ($instF && $instList && !in_array($instF, $instList, true)) continue;

            $items[] = [
                'url'      => $url,
                'obs'      => (string) ($it['observation_id'] ?? $it['observationId'] ?? ''),
                'program'  => (string) ($it['program'] ?? ''),
                'suffix'   => (string) ($it['details']['suffix'] ?? $it['suffix'] ?? ''),
                'inst'     => $instList,
                'caption'  => trim(
                    (($it['observation_id'] ?? '') ?: ($it['id'] ?? '')) .
                    ' · P' . ($it['program'] ?? '-') .
                    (($it['details']['suffix'] ?? '') ? ' · ' . $it['details']['suffix'] : '') .
                    ($instList ? ' · ' . implode('/', $instList) : '')
                ),
                'link'     => $loc ?: $url,
            ];
            if (count($items) >= $per) break;
        }

        return response()->json([
            'source' => $path,
            'count'  => count($items),
            'items'  => $items,
        ]);
    }

    // Сохраняем старый JSON endpoint для таблицы телеметрии (если зовут напрямую)
    public function data(\App\Http\Requests\DashboardDataRequest $request)
    {
        $query = DB::table('telemetry_legacy')
            ->select([
                'id',
                'recorded_at as created_at',
                'voltage as value',
                'temp',
                'flag_a',
                'flag_b',
                'count',
                'note',
                'source_file',
            ]);

        if ($request->filled('q')) {
            $term = $request->input('q');
            $query->where(function ($qb) use ($term) {
                $qb->where('note', 'ilike', "%{$term}%")
                    ->orWhere('source_file', 'ilike', "%{$term}%")
                    ->orWhereRaw("voltage::text ILIKE ?", ["%{$term}%"]);
            });
        }

        if ($request->filled('date_from')) {
            $query->where('recorded_at', '>=', $request->input('date_from'));
        }
        if ($request->filled('date_to')) {
            $query->where('recorded_at', '<=', $request->input('date_to'));
        }

        $allowedSorts = ['created_at', 'value', 'id', 'voltage', 'temp'];
        $sortBy = $request->input('sort_by', 'created_at');
        if (!in_array($sortBy, $allowedSorts)) {
            $sortBy = 'created_at';
        }
        $order = strtolower($request->input('order', 'desc')) === 'asc' ? 'asc' : 'desc';

        $perPage = (int) $request->input('per_page', 25);
        if ($perPage < 1) $perPage = 1;
        if ($perPage > 1000) $perPage = 1000;

        $page = max(1, (int) $request->input('page', 1));

        $dbSortCol = match ($sortBy) {
            'value' => 'voltage',
            'created_at' => 'recorded_at',
            default => $sortBy,
        };

        $total = (clone $query)->count();

        $rows = $query
            ->orderBy($dbSortCol, $order)
            ->offset(($page - 1) * $perPage)
            ->limit($perPage)
            ->get();

        return response()->json([
            'data' => $rows->toArray(),
            'meta' => [
                'page' => $page,
                'per_page' => $perPage,
                'total' => $total,
                'pages' => (int) ceil($total / max(1, $perPage)),
            ],
        ]);
    }
}
