@extends('layout')
@section('content')
<div class="container my-3">
  <h3 class="mb-3">{{ $title }}</h3>
  {{-- намеренно "плохо": сырое содержимое из БД --}}
  {!! $html !!}
</div>
@endsection
