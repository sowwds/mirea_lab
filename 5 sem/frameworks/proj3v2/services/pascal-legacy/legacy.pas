program LegacyCSV;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, DateUtils, Process, StrUtils;

function GetEnvDef(const name, def: string): string;
var v: string;
begin
  v := GetEnvironmentVariable(name);
  if v = '' then Exit(def) else Exit(v);
end;

function RandFloat(minV, maxV: Double): Double;
begin
  Result := minV + Random * (maxV - minV);
end;

function RandBool: Boolean;
begin
  Result := Random(2) = 1;
end;

function EscapeCSV(const s: string): string;
var
  t: string;
begin
  t := StringReplace(s, '"', '""', [rfReplaceAll]);
  if (Pos(',', t) > 0) or (Pos('"', t) > 0) or (Pos(#10, t) > 0) or (Pos(#13, t) > 0) then
    Result := '"' + t + '"'
  else
    Result := t;
end;

function DateTimeToUnixTimestamp(const dt: TDateTime): Int64;
var
  epoch: TDateTime;
begin
  epoch := EncodeDate(1970, 1, 1);
  Result := Round((dt - epoch) * 86400.0);
end;

function UnixToDateTime(ts: Int64): TDateTime;
var
  epoch: TDateTime;
begin
  epoch := EncodeDate(1970, 1, 1);
  Result := epoch + (ts / 86400.0);
end;

procedure WriteCSV(const fullpath: string; rows: Integer);
var
  f: TextFile;
  i: Integer;
  recorded_at_ts: Int64;
  recorded_at_dt: TDateTime;
  voltage, temp: Double;
  flagA, flagB: Boolean;
  note: string;
  countVal: Integer;
  fn: string;
begin
  fn := ExtractFileName(fullpath);
  AssignFile(f, fullpath);
  Rewrite(f);
  Writeln(f, 'recorded_at,flag_A,flag_B,voltage,temp,count,note,source_file');
  for i := 1 to rows do
  begin
    recorded_at_dt := IncSecond(Now, - (rows - i) * 60);
    recorded_at_ts := DateTimeToUnixTimestamp(recorded_at_dt);
    voltage := RandFloat(3.2, 12.6);
    temp := RandFloat(-50.0, 80.0);
    countVal := Random(1000);
    flagA := RandBool;
    flagB := RandBool;
    note := Format('Sample note %d', [i]);

    // Записываем timestamp как число, числа без кавычек, логические как ИСТИНА/ЛОЖЬ, строки в кавычках если нужно
    Writeln(f,
      IntToStr(recorded_at_ts) + ',' +
      IfThen(flagA, 'ИСТИНА', 'ЛОЖЬ') + ',' +
      IfThen(flagB, 'ИСТИНА', 'ЛОЖЬ') + ',' +
      FormatFloat('0.00', voltage) + ',' +
      FormatFloat('0.00', temp) + ',' +
      IntToStr(countVal) + ',' +
      EscapeCSV(note) + ',' +
      EscapeCSV(fn)
    );
  end;
  CloseFile(f);
end;

function ParseCSVLine(const line: string; out cols: TStringList): Boolean;
var
  i: Integer;
  inQuotes: Boolean;
  current: string;
  ch: Char;
begin
  Result := False;
  cols.Clear;
  inQuotes := False;
  current := '';
  for i := 1 to Length(line) do
  begin
    ch := line[i];
    if ch = '"' then
    begin
      if (i < Length(line)) and (line[i + 1] = '"') then
      begin
        current := current + '"';
        Inc(i);
      end
      else
        inQuotes := not inQuotes;
    end
    else if (ch = ',') and not inQuotes then
    begin
      cols.Add(current);
      current := '';
    end
    else
      current := current + ch;
  end;
  cols.Add(current);
  Result := True;
end;

function FormatTimestamp(ts: string): string;
var
  tsInt: Int64;
  dt: TDateTime;
begin
  if TryStrToInt64(ts, tsInt) then
  begin
    dt := UnixToDateTime(tsInt);
    Result := FormatDateTime('yyyy-mm-dd hh:nn:ss', dt);
  end
  else
    Result := ts;
end;

procedure CSVToHTML(const csvPath, htmlPath: string);
var
  sl: TStringList;
  outF: TextFile;
  i, j: Integer;
  cols: TStringList;
  headerCols: TStringList;
  line: string;
  cellValue: string;
  colName: string;
begin
  if not FileExists(csvPath) then Exit;
  sl := TStringList.Create;
  cols := TStringList.Create;
  headerCols := TStringList.Create;
  try
    sl.LoadFromFile(csvPath);
    AssignFile(outF, htmlPath);
    Rewrite(outF);
    Writeln(outF, '<!doctype html>');
    Writeln(outF, '<html><head>');
    Writeln(outF, '<meta charset="utf-8">');
    Writeln(outF, '<meta name="viewport" content="width=device-width, initial-scale=1">');
    Writeln(outF, '<title>' + ExtractFileName(csvPath) + '</title>');
    Writeln(outF, '<style>');
    Writeln(outF, 'body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }');
    Writeln(outF, '.container { max-width: 100%; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }');
    Writeln(outF, 'h2 { color: #333; margin-bottom: 20px; }');
    Writeln(outF, 'table { border-collapse: collapse; width: 100%; font-size: 14px; }');
    Writeln(outF, 'th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }');
    Writeln(outF, 'th { background: #4CAF50; color: white; font-weight: bold; position: sticky; top: 0; }');
    Writeln(outF, 'tr:nth-child(even) { background: #f9f9f9; }');
    Writeln(outF, 'tr:hover { background: #f1f1f1; }');
    Writeln(outF, '.number { text-align: right; font-family: monospace; }');
    Writeln(outF, '.timestamp { font-family: monospace; color: #666; }');
    Writeln(outF, '.boolean-true { color: #4CAF50; font-weight: bold; }');
    Writeln(outF, '.boolean-false { color: #f44336; font-weight: bold; }');
    Writeln(outF, '</style>');
    Writeln(outF, '</head><body>');
    Writeln(outF, '<div class="container">');
    Writeln(outF, '<h2>📊 Telemetry Data: ' + ExtractFileName(csvPath) + '</h2>');
    Writeln(outF, '<table>');
    if sl.Count > 0 then
    begin
      line := sl[0];
      ParseCSVLine(line, headerCols);
      Writeln(outF, '<thead><tr>');
      for j := 0 to headerCols.Count - 1 do
        Writeln(outF, '<th>' + headerCols[j] + '</th>');
      Writeln(outF, '</tr></thead>');
    end;
    Writeln(outF, '<tbody>');
    for i := 1 to sl.Count - 1 do
    begin
      line := sl[i];
      ParseCSVLine(line, cols);
      Writeln(outF, '<tr>');
      for j := 0 to cols.Count - 1 do
      begin
        cellValue := cols[j];
        colName := '';
        if j < headerCols.Count then
          colName := headerCols[j];
        
        // Apply formatting based on column name
        if colName = 'recorded_at' then
          cellValue := '<span class="timestamp">' + FormatTimestamp(cellValue) + '</span>'
        else if (colName = 'flag_A') or (colName = 'flag_B') then
        begin
          if cellValue = 'ИСТИНА' then
            cellValue := '<span class="boolean-true">' + cellValue + '</span>'
          else
            cellValue := '<span class="boolean-false">' + cellValue + '</span>';
        end
        else if (colName = 'voltage') or (colName = 'temp') or (colName = 'count') then
          cellValue := '<span class="number">' + cellValue + '</span>'
        else
          cellValue := StringReplace(StringReplace(cellValue, '<', '&lt;', [rfReplaceAll]), '>', '&gt;', [rfReplaceAll]);
        
        Writeln(outF, '<td>' + cellValue + '</td>');
      end;
      Writeln(outF, '</tr>');
    end;
    Writeln(outF, '</tbody></table>');
    Writeln(outF, '</div></body></html>');
    CloseFile(outF);
  finally
    headerCols.Free;
    cols.Free;
    sl.Free;
  end;
end;

procedure CreateExcelFallbackXML(const csvPath, xmlPath: string);
var
  sl: TStringList;
  outF: TextFile;
  i, j: Integer;
  cols: TStringList;
  line: string;
begin
  sl := TStringList.Create;
  cols := TStringList.Create;
  try
    sl.LoadFromFile(csvPath);
    AssignFile(outF, xmlPath);
    Rewrite(outF);
    Writeln(outF, '<?xml version="1.0"?>');
    Writeln(outF, '<?mso-application progid="Excel.Sheet"?>');
    Writeln(outF, '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"');
    Writeln(outF, ' xmlns:o="urn:schemas-microsoft-com:office:office"');
    Writeln(outF, ' xmlns:x="urn:schemas-microsoft-com:office:excel"');
    Writeln(outF, ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">');
    Writeln(outF, '<Worksheet ss:Name="Data">');
    Writeln(outF, '<Table>');
    for i := 0 to sl.Count - 1 do
    begin
      line := sl[i];
      cols.Delimiter := ',';
      cols.StrictDelimiter := True;
      cols.DelimitedText := line;
      Writeln(outF, '<Row>');
      for j := 0 to cols.Count - 1 do
      begin
        Writeln(outF, '<Cell><Data ss:Type="String">' + StringReplace(cols[j], '&', '&amp;', [rfReplaceAll]) + '</Data></Cell>');
      end;
      Writeln(outF, '</Row>');
    end;
    Writeln(outF, '</Table>');
    Writeln(outF, '</Worksheet>');
    Writeln(outF, '</Workbook>');
    CloseFile(outF);
  finally
    cols.Free;
    sl.Free;
  end;
end;

procedure CreateExcelFromCSV(const csvPath: string);
var
  xlsxPath, pythonScript, pythonCmd: string;
  outLines: TStringList;
  exitCode: Integer;
begin
  xlsxPath := ChangeFileExt(csvPath, '.xlsx');
  pythonScript := '/opt/legacy/csv_to_xlsx.py';
  
  // Используем Python скрипт для создания настоящего XLSX файла
  pythonCmd := 'python3 ' + pythonScript + ' ' + csvPath + ' ' + xlsxPath;
  exitCode := RunShellCommand(pythonCmd, outLines);
  
  if exitCode <> 0 then
  begin
    Writeln('Python XLSX conversion failed, falling back to XML format');
    Writeln(outLines.Text);
    // Fallback на старый XML формат
    CreateExcelFallbackXML(csvPath, xlsxPath);
  end
  else
    Writeln('[pascal] Generated XLSX with proper data types');
  
  outLines.Free;
end;

{ Run shell command using TProcess, capture stdout/stderr into TStringList and return exit code }
function RunShellCommand(const cmd: string; out outputLines: TStringList): Integer;
var
  Proc: TProcess;
  Buffer: array[0..2047] of byte;
  BytesRead: LongInt;
  MS: TMemoryStream;
  s: string;
begin
  outputLines := TStringList.Create;
  Proc := TProcess.Create(nil);
  MS := TMemoryStream.Create;
  try
    Proc.Executable := '/bin/sh';
    Proc.Parameters.Clear;
    Proc.Parameters.Add('-c');
    Proc.Parameters.Add(cmd);
    Proc.Options := Proc.Options + [poUsePipes, poWaitOnExit];
    Proc.ShowWindow := swoHIDE;
    try
      Proc.Execute;
    except
      on E: Exception do
      begin
        outputLines.Add('Execute error: ' + E.Message);
        Result := -1;
        Exit;
      end;
    end;

    // read stdout
    MS.Clear;
    while Proc.Output.NumBytesAvailable > 0 do
    begin
      BytesRead := Proc.Output.Read(Buffer, SizeOf(Buffer));
      if BytesRead > 0 then MS.Write(Buffer, BytesRead)
      else Break;
    end;

    if MS.Size > 0 then
    begin
      SetLength(s, MS.Size);
      MS.Position := 0;
      MS.ReadBuffer(s[1], MS.Size);
      outputLines.Text := s;
    end;

    // if no stdout, read stderr
    if (outputLines.Count = 0) and (Proc.Stderr.NumBytesAvailable > 0) then
    begin
      MS.Clear;
      while Proc.Stderr.NumBytesAvailable > 0 do
      begin
        BytesRead := Proc.Stderr.Read(Buffer, SizeOf(Buffer));
        if BytesRead > 0 then MS.Write(Buffer, BytesRead)
        else Break;
      end;
      if MS.Size > 0 then
      begin
        SetLength(s, MS.Size);
        MS.Position := 0;
        MS.ReadBuffer(s[1], MS.Size);
        outputLines.Text := s;
      end;
    end;

    Result := Proc.ExitStatus;
  finally
    Proc.Free;
    MS.Free;
  end;
end;

procedure GenerateAndCopy();
var
  outDir, fn, fullpath, pghost, pgport, pguser, pgpass, pgdb, copyCmd: string;
  rows: Integer;
  outLines: TStringList;
  exitCode: Integer;
begin
  outDir := GetEnvDef('CSV_OUT_DIR', '/data/csv');
  ForceDirectories(outDir);
  fn := 'telemetry_' + FormatDateTime('yyyymmdd_hhnnss', Now) + '.csv';
  fullpath := IncludeTrailingPathDelimiter(outDir) + fn;
  rows := StrToIntDef(GetEnvDef('ROWS_PER_FILE', '10'), 10);

  WriteCSV(fullpath, rows);
  CSVToHTML(fullpath, ChangeFileExt(fullpath, '.html'));
  CreateExcelFromCSV(fullpath);

  pghost := GetEnvDef('PGHOST', 'db');
  pgport := GetEnvDef('PGPORT', '5432');
  pguser := GetEnvDef('PGUSER', 'monouser');
  pgpass := GetEnvDef('PGPASSWORD', 'monopass');
  pgdb   := GetEnvDef('PGDATABASE', 'monolith');

  // Используем временную таблицу для импорта с конвертацией timestamp
  copyCmd := 'PGPASSWORD=' + pgpass + ' psql "host=' + pghost + ' port=' + pgport + ' user=' + pguser + ' dbname=' + pgdb + '" ' +
             '-c "CREATE TEMP TABLE temp_telemetry_import(recorded_at_ts BIGINT, flag_a_str TEXT, flag_b_str TEXT, voltage NUMERIC, temp NUMERIC, count INTEGER, note TEXT, source_file TEXT); ' +
             '\copy temp_telemetry_import(recorded_at_ts, flag_a_str, flag_b_str, voltage, temp, count, note, source_file) FROM ''' + fullpath + ''' WITH (FORMAT csv, HEADER true); ' +
             'INSERT INTO telemetry_legacy(recorded_at, flag_a, flag_b, voltage, temp, count, note, source_file) ' +
             'SELECT to_timestamp(recorded_at_ts), flag_a_str = ''ИСТИНА'', flag_b_str = ''ИСТИНА'', voltage, temp, count, note, source_file FROM temp_telemetry_import; ' +
             'DROP TABLE temp_telemetry_import;"';

  exitCode := RunShellCommand(copyCmd, outLines);
  if exitCode <> 0 then
  begin
    Writeln('COPY command exit code: ', exitCode);
    Writeln(outLines.Text);
  end
  else
    Writeln('[pascal] generated CSV + HTML preview + XLSX(xml) + imported to Postgres');
  outLines.Free;
end;

var
  period: Integer;
begin
  Randomize;
  period := StrToIntDef(GetEnvDef('GEN_PERIOD_SEC', '300'), 300);

  { generate one file immediately at start }
  try
    GenerateAndCopy();
  except
    on E: Exception do WriteLn('Startup generation error: ', E.Message);
  end;

  while True do
  begin
    Sleep(period * 1000);
    try
      GenerateAndCopy();
    except
      on E: Exception do WriteLn('Legacy error: ', E.Message);
    end;
  end;
end.
