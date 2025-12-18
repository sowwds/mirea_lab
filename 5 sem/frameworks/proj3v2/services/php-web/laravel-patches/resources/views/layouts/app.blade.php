<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Space Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/cyborg/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    :root {
      --bg-main: #101419;
      --bg-card: #1a222c;
      --border-color: rgba(136, 192, 208, 0.2);
      --accent-primary: #00aaff;
      --accent-secondary: #ff33aa;
      --text-main: #d0d8e0;
      --text-secondary: #8899a6;
      --glass-bg: rgba(26, 34, 44, 0.75);
    }

    body {
      min-height: 100vh;
      background-color: var(--bg-main);
      background-image: radial-gradient(circle at 1px 1px, rgba(136, 192, 208, 0.1) 1px, transparent 0);
      background-size: 20px 20px;
      color: var(--text-main);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .glass {
      background: var(--glass-bg);
      border: 1px solid var(--border-color);
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 12px;
    }

    .glass:hover {
      transform: translateY(-5px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
      border-color: rgba(136, 192, 208, 0.4);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border: none;
        box-shadow: 0 4px 20px rgba(0, 170, 255, 0.3);
        transition: all .25s ease;
        font-weight: 600;
    }

    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(255, 51, 170, 0.4);
    }
    
    .navbar {
      background: var(--glass-bg) !important;
      border-bottom: 1px solid var(--border-color);
      padding: 0.8rem 0;
    }

    a.nav-link {
        color: var(--text-secondary) !important;
        font-weight: 500;
        transition: all .2s ease;
        padding: 8px 16px;
        border-radius: 8px;
    }
    a.nav-link.active, a.nav-link:hover {
        color: var(--text-main) !important;
        background-color: rgba(136, 192, 208, 0.1);
    }
    .navbar-brand {
      color: var(--text-main) !important;
      font-weight: 700;
    }

    .table {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
    }

    .table thead th {
        background-color: #232E3B;
    }
    
    .table tbody tr {
      transition: all .15s ease-out;
    }

    .table tbody tr:hover {
      background-color: rgba(136, 192, 208, 0.05);
      transform: scale(1.02);
      box-shadow: 0 0 10px rgba(136, 192, 208, 0.1);
    }
    
    .card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
    }
    .card-header {
        background-color: #232E3B;
        border-bottom: 1px solid var(--border-color);
    }

    .form-control, .form-select {
        background-color: var(--bg-main);
        border: 1px solid var(--border-color);
        color: var(--text-main);
    }
    .form-control:focus, .form-select:focus {
        background-color: var(--bg-main);
        border-color: var(--accent-primary);
        box-shadow: 0 0 0 3px rgba(0, 170, 255, 0.25);
        color: var(--text-main);
    }

    #map {
        height: 340px; 
        border-radius: 12px; 
        border: 1px solid var(--border-color);
        filter: grayscale(80%) brightness(80%);
    }

    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(15px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
      animation: fadeInUp 0.5s ease-out forwards;
    }
  </style>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<nav class="navbar navbar-expand-lg mb-3 glass">
  <div class="container">
    <a class="navbar-brand fw-bold" href="/dashboard">Space Dashboard</a>
    <div class="d-flex gap-3">
      <a class="nav-link" href="/dashboard">Главная</a>
      <a class="nav-link" href="/iss">ISS</a>
      <a class="nav-link" href="/osdr">OSDR</a>
      <a class="nav-link" href="/telemetry">Telemetry</a>
      <a class="nav-link" href="/cms/page/demo">CMS</a>
    </div>
  </div>
</nav>
@yield('content')
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
