@extends('layouts.app')

@section('content')
<div class="container py-4">
  <h2>Страница из БД</h2>
  @if($page)
    <h4>{{ $page->title }}</h4>
    <div>{!! $page->body !!}</div>
  @else
    <div class="alert alert-warning">Страница не найдена</div>
  @endif
</div>
@endsection
