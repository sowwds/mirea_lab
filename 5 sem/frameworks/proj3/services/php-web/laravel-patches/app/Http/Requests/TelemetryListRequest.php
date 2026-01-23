<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class TelemetryListRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'search'    => ['nullable', 'string', 'max:200'],
            'keywords'  => ['nullable', 'string', 'max:200'],
            'from'      => ['nullable', 'date', 'before_or_equal:to'],
            'to'        => ['nullable', 'date', 'after_or_equal:from'],
            'flag_a'    => ['nullable', 'boolean'],
            'flag_b'    => ['nullable', 'boolean'],
            'sort'      => ['nullable', 'in:recorded_at,voltage,temp,count'],
            'dir'       => ['nullable', 'in:asc,desc'],
            'per_page'  => ['nullable', 'integer', 'min:1', 'max:200'],
        ];
    }

    public function messages(): array
    {
        return [
            'search.string' => 'Поисковый запрос должен быть строкой',
            'search.max' => 'Поисковый запрос не должен превышать 200 символов',
            'keywords.string' => 'Ключевые слова должны быть строкой',
            'keywords.max' => 'Ключевые слова не должны превышать 200 символов',
            'from.date' => 'Дата начала должна быть в формате даты',
            'from.before_or_equal' => 'Дата начала должна быть раньше или равна дате окончания',
            'to.date' => 'Дата окончания должна быть в формате даты',
            'to.after_or_equal' => 'Дата окончания должна быть позже или равна дате начала',
            'flag_a.boolean' => 'Флаг A должен быть логическим значением',
            'flag_b.boolean' => 'Флаг B должен быть логическим значением',
            'sort.in' => 'Поле сортировки должно быть одним из: recorded_at, voltage, temp, count',
            'dir.in' => 'Направление сортировки должно быть asc или desc',
            'per_page.integer' => 'Количество записей на странице должно быть числом',
            'per_page.min' => 'Минимальное количество записей на странице: 1',
            'per_page.max' => 'Максимальное количество записей на странице: 200',
        ];
    }
}

