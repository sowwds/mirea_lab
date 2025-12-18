<?php

namespace App\Http\Controllers;

use App\Http\Requests\AstroEventsRequest;

class AstroController extends Controller
{
    /** Убираем кавычки, фигурные скобки вида ${...} и пробелы */
    private function sanitizeCred(?string $v): string
    {
        $v = trim((string)$v, " \t\n\r\0\x0B\"'");
        if (str_starts_with($v, '${') && str_ends_with($v, '}')) {
            $v = substr($v, 2, -1);
        }
        return $v;
    }

    public function events(AstroEventsRequest $r)
    {

        $lat  = (float) $r->query('lat', 55.7558);
        $lon  = (float) $r->query('lon', 37.6176);
        $days = max(1, min(30, (int) $r->query('days', 7)));

        $from = now('UTC')->toDateString();
        $to   = now('UTC')->addDays($days)->toDateString();

        // Используем getenv() напрямую, так как env() может не работать в некоторых контекстах
        $appId  = $this->sanitizeCred(getenv('ASTRO_APP_ID') ?: env('ASTRO_APP_ID', ''));
        $secret = $this->sanitizeCred(getenv('ASTRO_APP_SECRET') ?: env('ASTRO_APP_SECRET', ''));
        
        if ($appId === '' || $secret === '') {
            return response()->json([
                'ok' => false,
                'error' => [
                    'code' => 'CONFIG_MISSING',
                    'message' => 'Missing ASTRO_APP_ID/ASTRO_APP_SECRET',
                ],
            ], 200);
        }

        $url  = 'https://api.astronomyapi.com/api/v2/bodies/events?' . http_build_query([
            'latitude'  => $lat,
            'longitude' => $lon,
            'from'      => $from,
            'to'        => $to,
        ]);

        // AstronomyAPI использует Basic Auth с appId:secret
        $credentials = trim($appId) . ':' . trim($secret);
        $authHeader = 'Basic ' . base64_encode($credentials);
        
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER     => [
                'Accept: application/json',
                'Authorization: ' . trim($authHeader),
                'User-Agent: monolith-iss/1.0'
            ],
            CURLOPT_TIMEOUT        => 25,
            CURLOPT_SSL_VERIFYPEER => true,
            CURLOPT_SSL_VERIFYHOST => 2,
            CURLOPT_VERBOSE        => false, // для отладки можно включить
        ]);
        $raw  = curl_exec($ch);
        $code = curl_getinfo($ch, CURLINFO_RESPONSE_CODE) ?: 0;
        $err  = curl_error($ch);
        curl_close($ch);

        if ($raw === false || $code >= 400) {
            return response()->json([
                'ok' => false,
                'error' => [
                    'code' => $code === 403 ? 'UPSTREAM_403' : 'UPSTREAM_ERROR',
                    'message' => $err ?: ("HTTP " . $code),
                    'status' => $code,
                ],
                'raw' => $raw,
            ], 200);
        }
        $json = json_decode($raw, true);
        return response()->json($json ?? ['raw' => $raw]);
    }
}
