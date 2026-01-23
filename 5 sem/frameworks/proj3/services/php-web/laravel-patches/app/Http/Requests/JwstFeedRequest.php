<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class JwstFeedRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'source'     => ['nullable', 'in:jpg,suffix,program'],
            'suffix'     => ['nullable', 'string', 'max:50'],
            'program'    => ['nullable', 'string', 'max:50'],
            'instrument' => ['nullable', 'string', 'max:50'],
            'page'       => ['nullable', 'integer', 'min:1', 'max:50'],
            'perPage'    => ['nullable', 'integer', 'min:1', 'max:60'],
        ];
    }

    public function messages(): array
    {
        return [
            'source.in' => 'Источник должен быть одним из: jpg, suffix, program',
            'suffix.string' => 'Суффикс должен быть строкой',
            'suffix.max' => 'Суффикс не должен превышать 50 символов',
            'program.string' => 'Программа должна быть строкой',
            'program.max' => 'Программа не должна превышать 50 символов',
            'instrument.string' => 'Инструмент должен быть строкой',
            'instrument.max' => 'Инструмент не должен превышать 50 символов',
            'page.integer' => 'Номер страницы должен быть числом',
            'page.min' => 'Минимальный номер страницы: 1',
            'page.max' => 'Максимальный номер страницы: 50',
            'perPage.integer' => 'Количество элементов на странице должно быть числом',
            'perPage.min' => 'Минимальное количество элементов: 1',
            'perPage.max' => 'Максимальное количество элементов: 60',
        ];
    }
}

