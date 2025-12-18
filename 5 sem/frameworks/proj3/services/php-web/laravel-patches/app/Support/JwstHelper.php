<?php

namespace App\Support;

final class JwstHelper
{
    private string $host;
    private string $key;
    private ?string $email;

    public function __construct()
    {
        $this->host  = rtrim(getenv('JWST_HOST') ?: 'https://api.jwstapi.com', '/');
        $this->key   = getenv('JWST_API_KEY') ?: '';
        $this->email = getenv('JWST_EMAIL') ?: null;
    }

    public function get(string $path, array $qs = []): array
    {
        $url = $this->host.'/'.ltrim($path, '/');
        if ($qs) $url .= (str_contains($url,'?')?'&':'?').http_build_query($qs);
        $headers = [
            'x-api-key: '.$this->key,
        ];
        if ($this->email) $headers[] = 'email: '.$this->email;

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 30,
            CURLOPT_HTTPHEADER => $headers,
        ]);
        $raw = curl_exec($ch);
        curl_close($ch);

        $j = json_decode((string)$raw, true);
        return is_array($j) ? $j : [];
    }

    /** ищем первую пригодную картинку в произвольной структуре */
    public static function pickImageUrl(array $v): ?string
    {
        $stack = [$v];
        while ($stack) {
            $cur = array_pop($stack);
            foreach ($cur as $k => $val) {
                if (is_string($val) && preg_match('~^https?://.*\.(?:jpg|jpeg|png)$~i', $val)) return $val;
                if (is_array($val)) $stack[] = $val;
            }
        }
        return null;
    }
}
