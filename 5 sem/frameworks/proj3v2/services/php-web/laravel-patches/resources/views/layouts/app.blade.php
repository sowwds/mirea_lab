<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Space Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/flatly/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    :root {
      --bg-main: #f8f9fa;
      --bg-card: #ffffff;
      --border-color: #dee2e6;
      --primary-accent: #2c3e50;
      --secondary-accent: #18bc9c;
      --text-main: #212529;
      --text-secondary: #6c757d;
    }

    body {
      min-height: 100vh;
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .card {
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 12px;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    
    .navbar {
      background-color: rgba(255, 255, 255, 0.9) !important;
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border-color);
      box-shadow: 0 2px 4px rgba(0,0,0,0.04);
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
        color: var(--primary-accent) !important;
        background-color: rgba(44, 62, 80, 0.05);
    }
    .navbar-brand {
      color: var(--primary-accent) !important;
      font-weight: 700;
    }

    .table {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border-color);
    }
    
    .table thead th {
        background-color: var(--bg-main);
    }
    
    .table tbody tr:hover {
      background-color: rgba(44, 62, 80, 0.04);
    }
    
    #map {
        height: 340px; 
        border-radius: 12px; 
        border: 1px solid var(--border-color);
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
