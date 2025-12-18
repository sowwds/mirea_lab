<?php
namespace App\Http\Controllers;
use Illuminate\Support\Facades\DB;

class CmsController extends Controller {
  public function page(string $slug) {
    $row = DB::selectOne("SELECT title, content FROM cms_blocks WHERE slug = ? AND is_active = TRUE", [$slug]);
    if (!$row) abort(404);
    return response()->view('cms.page', ['title' => $row->title, 'html' => $row->content]);
  }
}
