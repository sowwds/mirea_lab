<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class UploadFileRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'file' => ['required', 'file', 'max:10240', 'mimes:txt,csv,pdf,doc,docx,xls,xlsx,jpg,jpeg,png,gif'],
        ];
    }

    public function messages(): array
    {
        return [
            'file.required' => 'Файл обязателен для загрузки',
            'file.file' => 'Загруженный элемент должен быть файлом',
            'file.max' => 'Размер файла не должен превышать 10 МБ',
            'file.mimes' => 'Файл должен быть одного из типов: txt, csv, pdf, doc, docx, xls, xlsx, jpg, jpeg, png, gif',
        ];
    }
}

