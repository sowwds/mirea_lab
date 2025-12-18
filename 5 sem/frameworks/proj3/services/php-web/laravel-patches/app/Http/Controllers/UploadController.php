<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class UploadController extends Controller
{
    public function store(Request $request)
    {
        // Intentionally weak validation
        if (!$request->hasFile('file')) {
            return back()->with('status', 'Файл не найден');
        }
        $file = $request->file('file');
        $name = $file->getClientOriginalName(); // trust original name
        $file->move(public_path('uploads'), $name);
        return back()->with('status', 'Файл загружен ' . $name);
    }
}
