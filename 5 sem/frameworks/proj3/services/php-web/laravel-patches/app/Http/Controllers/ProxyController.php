<?php

namespace App\Http\Controllers;

use Illuminate\Http\Response;

class ProxyController extends Controller
{
    private function base(): string {
        return getenv('RUST_BASE') ?: 'http://rust_iss:3000';
    }

    public function last()  { return $this->pipe('/last'); }

    public function trend() {
        $q = request()->getQueryString();
        return $this->pipe('/iss/trend' . ($q ? '?' . $q : ''));
    }

    private function pipe(string $path)
    {
        $url = $this->base() . $path;
        try {
            $ctx = stream_context_create([
                'http' => ['timeout' => 5, 'ignore_errors' => true],
            ]);
            $body = @file_get_contents($url, false, $ctx);
            if ($body === false || trim($body) === '') {
                $body = '{}';
            }
            json_decode($body);
            if (json_last_error() !== JSON_ERROR_NONE) {
                $body = '{}';
            }
            return new Response($body, 200, ['Content-Type' => 'application/json']);
        } catch (\Throwable $e) {
            return new Response('{"error":"upstream"}', 200, ['Content-Type' => 'application/json']);
        }
    }
}
