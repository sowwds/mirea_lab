@extends('layouts.app')

@section('content')
<div class="container py-4">
  <h3 class="mb-3">МКС данные</h3>

  <div class="row g-3">
    <div class="col-md-6">
      <div class="card shadow-sm">
        <div class="card-body">
          <h5 class="card-title">Последний снимок</h5>
          @if(!empty($last['payload']))
            <ul class="list-group">
              <li class="list-group-item">Широта {{ $last['payload']['latitude'] ?? '—' }}</li>
              <li class="list-group-item">Долгота {{ $last['payload']['longitude'] ?? '—' }}</li>
              <li class="list-group-item">Высота км {{ $last['payload']['altitude'] ?? '—' }}</li>
              <li class="list-group-item">Скорость км/ч {{ $last['payload']['velocity'] ?? '—' }}</li>
              <li class="list-group-item">Время {{ $last['fetched_at'] ?? '—' }}</li>
            </ul>
          @else
            <div class="text-muted">нет данных</div>
          @endif
          <div class="mt-3"><code>{{ $base }}/last</code></div>
        </div>
      </div>
    </div>

    <div class="col-md-6">
      <div class="card shadow-sm">
        <div class="card-body">
          <h5 class="card-title">Тренд движения</h5>
          @if(!empty($trend))
            <ul class="list-group">
              <li class="list-group-item">Движение {{ ($trend['movement'] ?? false) ? 'да' : 'нет' }}</li>
              <li class="list-group-item">Смещение км {{ number_format($trend['delta_km'] ?? 0, 3, '.', ' ') }}</li>
              <li class="list-group-item">Интервал сек {{ $trend['dt_sec'] ?? 0 }}</li>
              <li class="list-group-item">Скорость км/ч {{ $trend['velocity_kmh'] ?? '—' }}</li>
            </ul>
          @else
            <div class="text-muted">нет данных</div>
          @endif
          <div class="mt-3"><code>{{ $base }}/iss/trend</code></div>
          <div class="mt-3"><a class="btn btn-outline-primary" href="/osdr">Перейти к OSDR</a></div>
        </div>
      </div>
    </div>
  </div>
</div>
@endsection
