<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class DashboardDataRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'q'         => ['nullable', 'string', 'max:200'],
            'date_from' => ['nullable', 'date', 'before_or_equal:date_to'],
            'date_to'   => ['nullable', 'date', 'after_or_equal:date_from'],
            'sort_by'   => ['nullable', 'in:created_at,value,id,voltage,temp'],
            'order'     => ['nullable', 'in:asc,desc'],
            'per_page'  => ['nullable', 'integer', 'min:1', 'max:1000'],
            'page'      => ['nullable', 'integer', 'min:1'],
        ];
    }

    public function messages(): array
    {
        return [
            'q.string' => 'Поисковый запрос должен быть строкой',
            'q.max' => 'Поисковый запрос не должен превышать 200 символов',
            'date_from.date' => 'Дата начала должна быть в формате даты',
            'date_from.before_or_equal' => 'Дата начала должна быть раньше или равна дате окончания',
            'date_to.date' => 'Дата окончания должна быть в формате даты',
            'date_to.after_or_equal' => 'Дата окончания должна быть позже или равна дате начала',
            'sort_by.in' => 'Поле сортировки должно быть одним из: created_at, value, id, voltage, temp',
            'order.in' => 'Направление сортировки должно быть asc или desc',
            'per_page.integer' => 'Количество записей на странице должно быть числом',
            'per_page.min' => 'Минимальное количество записей на странице: 1',
            'per_page.max' => 'Максимальное количество записей на странице: 1000',
            'page.integer' => 'Номер страницы должен быть числом',
            'page.min' => 'Минимальный номер страницы: 1',
        ];
    }
}

