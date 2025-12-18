<?php
namespace App\Http\Controllers;

final class JwstHelper
{
    private string $host;
    private string $key;
    private ?string $email;

    public function __construct()
    {
        $this->host  = rtrim(getenv('JWST_HOST') ?: 'https://api.jwstapi.com','/');
        $this->key   = getenv('JWST_API_KEY') ?: '';
        $this->email = getenv('JWST_EMAIL') ?: null;
    }

    public function get(string $path, array $qs = []): array
    {
        $url = $this->host . '/' . ltrim($path,'/');
        if ($qs) $url .= (str_contains($url,'?') ? '&' : '?') . http_build_query($qs);

        $headers = "x-api-key: {$this->key}\r\n";
        if ($this->email) $headers .= "email: {$this->email}\r\n";

        $ctx = stream_context_create(['http'=>[
            'method'=>'GET','timeout'=>12,'ignore_errors'=>true,
            'header'=>$headers
        ]]);
        $raw = @file_get_contents($url,false,$ctx);
        $json = $raw ? json_decode($raw,true) : null;
        return is_array($json) ? $json : [];
    }

    public static function pickImageUrl(array $item): ?string
    {
        $keys = ['thumbnail','thumbnailUrl','image','img','url','href','link','s3_url','file_url'];
        foreach ($keys as $k) {
            $v = $item[$k] ?? null;
            if (is_string($v)) {
                $u = trim($v);
                if (preg_match('~^https?://~i',$u) && preg_match('~\.(jpg|jpeg|png)$~i',$u)) return $u;
                if (str_starts_with($u,'/') && preg_match('~\.(jpg|jpeg|png)$~i',$u)) return 'https://api.jwstapi.com'.$u;
            }
        }
        foreach ($item as $v) if (is_array($v)) { $u = self::pickImageUrl($v); if ($u) return $u; }
        return null;
    }
}
