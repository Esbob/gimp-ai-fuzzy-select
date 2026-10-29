; The Windows setup for AI Fuzzy Select (Inno Setup 6.7; build: ISCC windows\setup.iss, with
; /DAppVersion=1.4.0 for a release). It checks what's needed (GIMP 3, Python, an NVIDIA
; graphics card, disk space), installs Python from python.org if there's none, then runs
; install_ai_select.py --progress, which downloads and tests everything, and shows its
; progress. Installs for the current user only; no administrator rights needed.
; Options: /CPU (the AI on the processor even with an NVIDIA graphics card), /MODEL=large
; (SAM 2.1 large and small) or /MODEL=small (small only), and Inno Setup's own, e.g. /SILENT.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
; The Python the setup installs when there's none (python.org's Windows installer)
#define PyVersion "3.14.10"
#define PySha256 "92480239264744fe0386a1bd8275786fefa126e3129e56a68e22b52c73329965"
#define AppId "F5CCDB8D-C469-4EBB-BAA4-5BB1F6E3EE63"

[Setup]
AppId={{{#AppId}}
AppName=AI Fuzzy Select for GIMP
AppVersion={#AppVersion}
AppPublisher=Esben Visfeldt
AppPublisherURL=https://github.com/Esbob/gimp-ai-fuzzy-select
AppSupportURL=https://github.com/Esbob/gimp-ai-fuzzy-select/issues
DefaultDirName={userpf}\AI Fuzzy Select
DisableWelcomePage=no
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; Not Windows' RedirectionGuard, which Inno Setup turns on by default and the programs Setup
; starts inherit: it protects installers that run as administrator, while this one runs as the
; user, in the user's own folders. With it, pip in install_ai_select.py failed (WinError 448)
; on a junction on the PATH that a tool had made for the user (uv's Python, here).
RedirectionGuard=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern dynamic
WizardImageFile=wizard-202.png,wizard-269.png,wizard-336.png,wizard-430.png
WizardSmallImageFile=wizard-small-58.png,wizard-small-77.png,wizard-small-97.png,wizard-small-124.png
OutputDir=output
OutputBaseFilename=gimp-ai-fuzzy-select-setup
Compression=lzma2/max
SolidCompression=yes
SetupLogging=yes
UninstallDisplayName=AI Fuzzy Select for GIMP
CloseApplications=no

[Messages]
WelcomeLabel1=Welcome to AI Fuzzy Select for GIMP
WelcomeLabel2=The AI Fuzzy Select plug-in adds a new menu item to GIMP's Select menu called "AI Fuzzy Select...": Click an object in an image and the AI selects it.%n%nSetup checks what it needs, downloads the AI (up to 4.5 GB) and tests it in GIMP. It installs for you only; no administrator rights are needed.
ReadyLabel2a=Click Install to start. Setup downloads the AI, then tests it in GIMP.
FinishedHeadingLabel=AI Fuzzy Select is installed
FinishedLabelNoIcons=It has been tested and works.%n%nIn GIMP (restart it if it's open): Select > AI Fuzzy Select...

[Files]
Source: "..\install_ai_select.py"; DestDir: "{app}"
Source: "..\uninstall_ai_select.py"; DestDir: "{app}"
Source: "..\gimp_ai_select\*.py"; DestDir: "{app}\gimp_ai_select"
Source: "..\LICENSE"; DestDir: "{app}"
Source: "ok.png"; Flags: dontcopy
Source: "missing.png"; Flags: dontcopy
Source: "install.png"; Flags: dontcopy
Source: "info.png"; Flags: dontcopy
Source: "warn.png"; Flags: dontcopy

[Run]
Filename: "{code:GimpExe}"; Description: "Start GIMP"; Flags: postinstall nowait skipifsilent; Check: CanStartGimp

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
const
  ROWS = 5;
  PythonVersions = '3.14 3.13 3.12 3.11 3.10';

var
  ReqPage: TWizardPage;
  ModelPage: TInputOptionWizardPage;
  RowIcon: array[0..ROWS - 1] of TBitmapImage;
  RowTitle, RowText: array[0..ROWS - 1] of TNewStaticText;
  CheckButton: TNewButton;
  { what was found }
  GimpExePath, GimpName, GimpProfile: String;
  PythonExe, PythonVersion: String;
  GpuName, GpuDriver, SmiLine: String;
  GpuMemMB: Integer;
  GpuCap: Extended;
  HasNvidia, Upgrade, ReadyToGo, ModelChosen, GimpHasPython: Boolean;
  PythonBySetup, LauncherBefore: Boolean;
  FreeMB, NeedMB, DownloadMB: Integer;
  { the install }
  PctLabel: TNewStaticText;
  DetailsButton: TNewButton;
  DetailsMemo: TNewMemo;
  Running, Stopping: Boolean;
  PyPid, Base: Integer;
  Outcome, ErrorText: String;
  InstalledKB: Integer;

{ --- Finding what's needed ------------------------------------------------------------ }

const
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall';

{ The key under HKCU's uninstall entries of the program whose name starts with Name, or ''.
  Bundle: only one with its own uninstaller (Python's setup, not its parts, which are MSIs). }
function FindUninstallKey(const Name: String; Bundle: Boolean): String;
var
  Keys: TArrayOfString;
  I: Integer;
  Display, Cmd: String;
begin
  Result := '';
  if RegGetSubkeyNames(HKCU, UninstallKey, Keys) then
    for I := 0 to GetArrayLength(Keys) - 1 do
      if (Result = '') and RegQueryStringValue(HKCU, UninstallKey + '\' + Keys[I], 'DisplayName', Display)
         and (Pos(Name, Display) = 1) then begin
        Cmd := '';
        RegQueryStringValue(HKCU, UninstallKey + '\' + Keys[I], 'UninstallString', Cmd);
        if not Bundle or (Pos('/uninstall', Lowercase(Cmd)) > 0) then
          Result := Keys[I];
      end;
end;

function NewestGimpProfile: String;
var
  Find: TFindRec;
  Root: String;
  Minor, Best: Integer;
begin
  Result := '';
  Best := -1;
  Root := ExpandConstant('{userappdata}\GIMP\');
  if FindFirst(Root + '3.*', Find) then
    try
      repeat
        Minor := StrToIntDef(Copy(Find.Name, 3, 5), -1);
        if (Find.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0) and (Minor > Best) then begin
          Best := Minor;
          Result := Root + Find.Name;
        end;
      until not FindNext(Find);
    finally
      FindClose(Find);
    end;
end;

procedure FindGimp;
var
  Key, Icon, Name: String;
  RootKeys: array of Integer;
  I: Integer;
begin
  GimpExePath := '';
  GimpName := '';
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\GIMP-3_is1';
  RootKeys := [HKCU, HKLM64, HKLM32];
  for I := 0 to GetArrayLength(RootKeys) - 1 do
    if (GimpExePath = '') and RegQueryStringValue(RootKeys[I], Key, 'DisplayIcon', Icon)
       and FileExists(RemoveQuotes(Icon)) then begin
      GimpExePath := RemoveQuotes(Icon);
      if RegQueryStringValue(RootKeys[I], Key, 'DisplayName', Name) then
        GimpName := Name;
    end;
  if GimpExePath = '' then
    if FileExists(ExpandConstant('{localappdata}\Programs\GIMP 3\bin\gimp-3.exe')) then
      GimpExePath := ExpandConstant('{localappdata}\Programs\GIMP 3\bin\gimp-3.exe')
    else if FileExists(ExpandConstant('{commonpf64}\GIMP 3\bin\gimp-3.exe')) then
      GimpExePath := ExpandConstant('{commonpf64}\GIMP 3\bin\gimp-3.exe');
  if GimpName = '' then
    GimpName := 'GIMP 3';
  GimpProfile := NewestGimpProfile;
end;

{ A python.org Python of a supported version (PEP 514 registry entries), not the Microsoft
  Store's: Windows keeps the files it creates under AppData apart, where GIMP doesn't look. }
procedure FindPython;
var
  Versions: TStringList;
  RootKeys: array of Integer;
  I, J: Integer;
  Path: String;
begin
  PythonExe := '';
  Versions := TStringList.Create;
  try
    Versions.Delimiter := ' ';
    Versions.DelimitedText := PythonVersions;
    RootKeys := [HKCU, HKLM64];
    for I := 0 to Versions.Count - 1 do
      for J := 0 to GetArrayLength(RootKeys) - 1 do
        if (PythonExe = '') and RegQueryStringValue(RootKeys[J],
             'Software\Python\PythonCore\' + Versions[I] + '\InstallPath', '', Path) then begin
          Path := AddBackslash(Path) + 'python.exe';
          if FileExists(Path) and (Pos('WindowsApps', Path) = 0)
             and (Pos('PythonSoftwareFoundation', Path) = 0) then begin
            PythonExe := Path;
            PythonVersion := Versions[I];
          end;
        end;
  finally
    Versions.Free;
  end;
#ifdef TestPythonInstall
  { a test build (ISCC /DTestPythonInstall): install Python as if there were none }
  if not Running then
    PythonExe := '';
#endif
end;

procedure OnSmiLine(const S: String; const Error, FirstLine: Boolean);
begin
  if FirstLine then
    SmiLine := S;
end;

{ nvidia-smi's "memory, driver, compute capability, name" (name last: it may hold commas) }
function HasSwitch(const Name: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), Name) = 0 then
      Result := True;
end;

procedure FindNvidia;
var
  Code, P: Integer;
  Rest: String;
begin
  HasNvidia := False;
  SmiLine := '';
  if HasSwitch('/CPU') then  { the AI on the processor even with an NVIDIA graphics card }
    exit;
  try
    ExecAndLogOutput('nvidia-smi.exe', '--query-gpu=memory.total,driver_version,compute_cap,name '
      + '--format=csv,noheader,nounits', '', SW_HIDE, ewWaitUntilTerminated, Code, @OnSmiLine);
  except
    Code := -1;
  end;
  if (Code <> 0) or (SmiLine = '') then
    exit;
  Rest := SmiLine;
  P := Pos(',', Rest);
  GpuMemMB := StrToIntDef(Trim(Copy(Rest, 1, P - 1)), 0);
  Rest := Copy(Rest, P + 1, MaxInt);
  P := Pos(',', Rest);
  GpuDriver := Trim(Copy(Rest, 1, P - 1));
  Rest := Copy(Rest, P + 1, MaxInt);
  P := Pos(',', Rest);
  GpuCap := StrToFloat(Trim(Copy(Rest, 1, P - 1)));
  GpuName := Trim(Copy(Rest, P + 1, MaxInt));
  HasNvidia := True;
end;

function DriverMajor: Integer;
begin
  Result := StrToIntDef(Copy(GpuDriver, 1, Pos('.', GpuDriver + '.') - 1), 0);
end;

function DataDir(const Sub: String): String;
begin
  Result := ExpandConstant('{localappdata}\gimp-ai-select\') + Sub;
end;

{ An earlier installation's virtual environment that still works: the Python it was made with
  (pyvenv.cfg's "home") is still there. (If not, install_ai_select.py makes a new one.) }
function VenvWorks: Boolean;
var
  Lines: TArrayOfString;
  I: Integer;
begin
  Result := False;
  if LoadStringsFromFile(DataDir('venv\pyvenv.cfg'), Lines) then
    for I := 0 to GetArrayLength(Lines) - 1 do
      if Pos('home', Trim(Lines[I])) = 1 then
        Result := FileExists(AddBackslash(Trim(Copy(Lines[I], Pos('=', Lines[I]) + 1, MaxInt)))
          + 'python.exe');
end;

{ The PyTorch build install_ai_select.py installs: its CUDA version as in torch/version.py }
function TorchCuda: String;
begin
  if not HasNvidia then
    Result := 'None'
  else if (GpuCap >= 7.5) and (DriverMajor >= 580) then
    Result := '''13.0'''
  else
    Result := '''12.6''';
end;

{ The download in MB, as install_ai_select.py will do it (see TORCH_MB and MODELS there) }
procedure Estimate;
var
  Large: Boolean;
  Version: AnsiString;
begin
  Large := ModelPage.SelectedValueIndex = 0;
  Upgrade := VenvWorks;
  DownloadMB := 0;
  if not Upgrade or not LoadStringFromFile(DataDir('venv\Lib\site-packages\torch\version.py'), Version)
     or (Pos('cuda: Optional[str] = ' + TorchCuda, Version) = 0) then
    if TorchCuda = 'None' then
      DownloadMB := DownloadMB + 130
    else if TorchCuda = '''13.0''' then
      DownloadMB := DownloadMB + 2000
    else
      DownloadMB := DownloadMB + 2630;
  if not Upgrade then
    DownloadMB := DownloadMB + 150;  { the smaller packages, SAM 2's source }
  if not FileExists(DataDir('checkpoints\sam2.1_hiera_small.pt')) then
    DownloadMB := DownloadMB + 185;
  if Large and not FileExists(DataDir('checkpoints\sam2.1_hiera_large.pt')) then
    DownloadMB := DownloadMB + 900;
  if not FileExists(DataDir('checkpoints\big-lama.pt')) then
    DownloadMB := DownloadMB + 206;
  if not FileExists(DataDir('checkpoints\ViTMatte_B_DIS.pth')) then
    DownloadMB := DownloadMB + 387;
  { what the install takes, plus room for the largest download while it's unpacked }
  if Upgrade then
    NeedMB := DownloadMB * 2 + 500
  else if HasNvidia then
    NeedMB := 9000
  else
    NeedMB := 4000;
end;

function SizeText(MB: Integer): String;
begin
  if MB >= 1000 then
    Result := Format('%.1f GB', [MB / 1000.0])
  else
    Result := Format('%d MB', [MB]);
end;

{ --- The model page -------------------------------------------------------------------- }

procedure CreateModelPage;
begin
  ModelPage := CreateInputOptionPage(ReqPage.ID, 'Select AI models', '', 'Which models should Setup download?',
    True, False);
  ModelPage.Add('');  { 0: large and small }
  ModelPage.Add('');  { 1: small only }
end;

{ The texts for the graphics card found, and the choice install_ai_select.py would make (large
  with an NVIDIA graphics card of 3 GB or more), until the user has chosen }
procedure UpdateModelPage;
var
  Big: Boolean;
  Both, Small: String;
begin
  Big := HasNvidia and (GpuMemMB >= 3000);
  Both := 'Large and small';
  Small := 'Small only';
  if Big then
    Both := Both + ' (recommended)'
  else
    Small := Small + ' (recommended)';
  ModelPage.Description := 'The AI that finds the object you click is SAM 2.1, Meta''s Segment '
    + 'Anything Model. It comes in two sizes.';
  if HasNvidia then
    Both := Both + ': the best selections. If the large model doesn''t fit in the memory of your '
      + 'graphics card, the small model will automatically be used. 1.1 GB'
  else begin
    ModelPage.Description := ModelPage.Description + ' Without an NVIDIA graphics card, it runs '
      + 'on the processor.';
    Both := Both + ': the best selections, but large is slow on the processor. 1.1 GB';
  end;
  ModelPage.CheckListBox.ItemCaption[0] := Both;
  ModelPage.CheckListBox.ItemCaption[1] := Small + ': quicker, but less accurate. 185 MB';
  if not ModelChosen then
    if ExpandConstant('{param:MODEL|}') = 'large' then
      ModelPage.SelectedValueIndex := 0
    else if ExpandConstant('{param:MODEL|}') = 'small' then
      ModelPage.SelectedValueIndex := 1
    else if Big then
      ModelPage.SelectedValueIndex := 0
    else
      ModelPage.SelectedValueIndex := 1;
end;

{ --- The requirements page ------------------------------------------------------------ }

procedure SetRow(Row: Integer; const Icon, Title, Text: String);
begin
  RowIcon[Row].PngImage.LoadFromFile(ExpandConstant('{tmp}\') + Icon + '.png');
  RowIcon[Row].Visible := True;
  RowTitle[Row].Caption := Title;
  RowTitle[Row].Visible := True;
  RowText[Row].Caption := Text;
  RowText[Row].Visible := True;
end;

procedure CheckRequirements;
var
  Free, Total: Int64;
  Drive, Space: String;
begin
  WizardForm.NextButton.Enabled := False;
  FindGimp;
  FindPython;
  FindNvidia;
  UpdateModelPage;
  Estimate;
  ReadyToGo := True;

  { GIMP's Python is a part of GIMP's installer that can be left out (the Compact install):
    then GIMP runs no Python plug-ins }
  GimpHasPython := (GimpExePath = '') or FileExists(ExtractFilePath(GimpExePath) + 'python.exe');
#ifdef TestNoGimpPython
  GimpHasPython := False;  { (a test build: ISCC /DTestNoGimpPython) }
#endif
  if not GimpHasPython then begin
    ReadyToGo := False;
    SetRow(0, 'missing', 'GIMP 3.x', GimpName + ' is installed without its Python support, which AI '
      + 'Fuzzy Select needs. Run GIMP''s installer again and choose Full (or Custom with Python '
      + 'plug-in support), then click Check again.');
  end else if GimpProfile <> '' then
    SetRow(0, 'ok', 'GIMP 3.x', 'Found: ' + GimpName + '.')
  else begin
    ReadyToGo := False;
    if GimpExePath <> '' then
      SetRow(0, 'missing', 'GIMP 3.x', GimpName + ' is installed, but hasn''t been started yet. Start '
        + 'GIMP once and close it, then click Check again.')
    else
      SetRow(0, 'missing', 'GIMP 3.x', 'Not found. Install GIMP 3 from gimp.org, start it once and '
        + 'close it, then click Check again.');
  end;

  if PythonExe <> '' then
    SetRow(1, 'ok', 'Python 3.10 to 3.14', 'Found: Python ' + PythonVersion + '.')
  else
    SetRow(1, 'install', 'Python 3.10 to 3.14', 'Not found: Setup installs Python {#PyVersion} from '
      + 'python.org, for your user only (32 MB).');

  if not HasNvidia then
    SetRow(2, 'info', 'NVIDIA graphics card (optional)', 'Not found: the AI runs on the processor, '
      + 'which is slower when opening an image.')
  else if DriverMajor < 528 then
    SetRow(2, 'warn', 'NVIDIA graphics card (optional)', GpuName + ': its driver (' + GpuDriver
      + ') is too old for the AI. Update it from nvidia.com, or the AI runs on the processor.')
  else
    SetRow(2, 'ok', 'NVIDIA graphics card (optional)', 'Found: ' + GpuName + ' ('
      + IntToStr((GpuMemMB + 512) div 1024) + ' GB). The AI runs on it.');

  FreeMB := 0;
  Drive := ExtractFileDrive(ExpandConstant('{localappdata}'));
  if GetSpaceOnDisk64(Drive, Free, Total) then
    FreeMB := Free div (1024 * 1024);
  Space := SizeText(FreeMB) + ' free on ' + Drive + '; about ' + SizeText(NeedMB) + ' needed.';
  if FreeMB >= NeedMB then
    SetRow(3, 'ok', 'Disk space', Space)
  else begin
    ReadyToGo := False;
    SetRow(3, 'missing', 'Disk space', 'Only ' + Space + ' Free up some space, then click Check again.');
  end;

  if Upgrade then
    SetRow(4, 'info', 'Earlier installation', 'Found: it will be updated, reusing what is '
      + 'already downloaded.');

  WizardForm.NextButton.Enabled := ReadyToGo;
end;

procedure CheckButtonClick(Sender: TObject);
begin
  CheckRequirements;
end;

procedure CreateRequirementsPage;
var
  I, Y: Integer;
begin
  ReqPage := CreateCustomPage(wpWelcome, 'Before you start', 'What AI Fuzzy Select needs.');
  Y := 0;
  for I := 0 to ROWS - 1 do begin
    RowIcon[I] := TBitmapImage.Create(ReqPage);
    RowIcon[I].Parent := ReqPage.Surface;
    RowIcon[I].SetBounds(0, Y + ScaleY(2), ScaleX(24), ScaleY(24));
    RowIcon[I].Stretch := True;
    RowIcon[I].BackColor := clNone;  { (else the PNG's transparent corners show button grey) }
    RowIcon[I].Visible := False;
    RowTitle[I] := TNewStaticText.Create(ReqPage);
    RowTitle[I].Parent := ReqPage.Surface;
    RowTitle[I].SetBounds(ScaleX(36), Y, ReqPage.SurfaceWidth - ScaleX(36), ScaleY(16));
    RowTitle[I].Font.Style := [fsBold];
    RowTitle[I].Visible := False;
    RowText[I] := TNewStaticText.Create(ReqPage);
    RowText[I].Parent := ReqPage.Surface;
    RowText[I].AutoSize := False;
    RowText[I].WordWrap := True;
    RowText[I].SetBounds(ScaleX(36), Y + ScaleY(17), ReqPage.SurfaceWidth - ScaleX(36), ScaleY(31));
    RowText[I].Visible := False;
    Y := Y + ScaleY(50);
  end;
  CheckButton := TNewButton.Create(ReqPage);
  CheckButton.Parent := ReqPage.Surface;
  CheckButton.Caption := 'Check again';
  CheckButton.SetBounds(0, Y, ScaleX(100), WizardForm.NextButton.Height);
  CheckButton.OnClick := @CheckButtonClick;
end;

{ --- Installing ----------------------------------------------------------------------- }

procedure SetPercent(PerMille: Integer);
begin
  WizardForm.ProgressGauge.Max := 1000;
  WizardForm.ProgressGauge.Position := PerMille;
  PctLabel.Caption := IntToStr(PerMille div 10) + '%';
end;

procedure DetailsButtonClick(Sender: TObject);
begin
  DetailsMemo.Visible := not DetailsMemo.Visible;
  if DetailsMemo.Visible then begin
    DetailsButton.Caption := 'Hide details';
    SendMessage(DetailsMemo.Handle, $00B6 { EM_LINESCROLL }, 0, DetailsMemo.Lines.Count);
  end
  else
    DetailsButton.Caption := 'Show details';
end;

procedure AddDetail(const S: String);
begin
  DetailsMemo.Lines.Add(S);
  if DetailsMemo.Visible then
    SendMessage(DetailsMemo.Handle, $00B6 { EM_LINESCROLL }, 0, 1);  { keep the last line in view }
end;

procedure CreateInstallingControls;
var
  Page: TNewNotebookPage;
  Top: Integer;
begin
  Page := WizardForm.InstallingPage;
  Top := WizardForm.ProgressGauge.Top + WizardForm.ProgressGauge.Height + ScaleY(6);
  PctLabel := TNewStaticText.Create(WizardForm);
  PctLabel.Parent := Page;
  PctLabel.Top := Top;
  PctLabel.Left := WizardForm.ProgressGauge.Left;
  PctLabel.Caption := '';
  DetailsButton := TNewButton.Create(WizardForm);
  DetailsButton.Parent := Page;
  DetailsButton.Caption := 'Show details';
  DetailsButton.SetBounds(WizardForm.ProgressGauge.Left + WizardForm.ProgressGauge.Width - ScaleX(100),
    Top, ScaleX(100), WizardForm.NextButton.Height);
  DetailsButton.OnClick := @DetailsButtonClick;
  DetailsMemo := TNewMemo.Create(WizardForm);
  DetailsMemo.Parent := Page;
  DetailsMemo.ReadOnly := True;
  DetailsMemo.ScrollBars := ssVertical;
  DetailsMemo.WordWrap := False;
  DetailsMemo.SetBounds(WizardForm.ProgressGauge.Left, Top + DetailsButton.Height + ScaleY(8),
    WizardForm.ProgressGauge.Width, Page.Height - Top - DetailsButton.Height - ScaleY(8));
  DetailsMemo.Visible := False;
end;

function EnableWindow(Wnd: HWND; Enable: BOOL): BOOL; external 'EnableWindow@user32.dll stdcall';

{ While Setup waits for a program it disables its window: enabled again, Cancel and Show
  details can be clicked during the install (Setup still handles messages meanwhile). }
procedure KeepClickable;
begin
  EnableWindow(WizardForm.Handle, True);
end;

function OnPythonDownload(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  KeepClickable;
  if ProgressMax > 0 then
    SetPercent(Progress * 20 div ProgressMax);
  WizardForm.FilenameLabel.Caption := Format('%d of %d MB', [Progress div 1000000, ProgressMax div 1000000]);
  Result := not Stopping;
end;

function InstallPython: Boolean;
var
  Code: Integer;
begin
  Result := False;
  WizardForm.StatusLabel.Caption := 'Downloading Python {#PyVersion} (32 MB)';
  AddDetail('Downloading Python {#PyVersion} from python.org');
  try
    DownloadTemporaryFile('https://www.python.org/ftp/python/{#PyVersion}/python-{#PyVersion}-amd64.exe',
      'python-setup.exe', '{#PySha256}', @OnPythonDownload);
  except
    if not Stopping then
      ErrorText := 'Python could not be downloaded from python.org: ' + GetExceptionMessage;
    exit;
  end;
  WizardForm.StatusLabel.Caption := 'Installing Python {#PyVersion} for your user';
  { (also py.exe without an uninstall entry: uninstalling Python can leave it behind) }
  LauncherBefore := (FindUninstallKey('Python Launcher', False) <> '')
    or FileExists(ExpandConstant('{localappdata}\Programs\Python\Launcher\py.exe'))
    or FileExists(ExpandConstant('{win}\py.exe'));
  WizardForm.FilenameLabel.Caption := '';
  SetPercent(25);
  if not Exec(ExpandConstant('{tmp}\python-setup.exe'), '/quiet InstallAllUsers=0 Include_launcher=1 '
      + 'InstallLauncherAllUsers=0 PrependPath=0 AssociateFiles=0 Include_test=0 Shortcuts=0',
      '', SW_SHOW, ewWaitUntilTerminated, Code) or (Code <> 0) then begin
    ErrorText := Format('The Python installer failed (exit code %d).', [Code]);
    exit;
  end;
  FindPython;
  if PythonExe = '' then begin
    ErrorText := 'Python was installed, but Setup can''t find it.';
    exit;
  end;
  SetPercent(40);
  PythonBySetup := True;
  Result := True;
end;

{ Lines from install_ai_select.py --progress: "@@progress <percent>\t<what>\t<detail>",
  "@@pid", "@@error", "@@result", "@@size"; everything else goes to the details and the log. }
procedure OnInstallerLine(const S: String; const Error, FirstLine: Boolean);
var
  Rest, Pct: String;
  P: Integer;
begin
  KeepClickable;
  if Copy(S, 1, 11) = '@@progress ' then begin
    Rest := Copy(S, 12, MaxInt);
    P := Pos(#9, Rest);
    Pct := Copy(Rest, 1, P - 1);
    StringChangeEx(Pct, '.', '', True);
    SetPercent(Base + StrToIntDef(Pct, 0) * (1000 - Base) div 1000);
    Rest := Copy(Rest, P + 1, MaxInt);
    P := Pos(#9, Rest);
    if WizardForm.StatusLabel.Caption <> Copy(Rest, 1, P - 1) then
      Log(PctLabel.Caption + ' ' + Copy(Rest, 1, P - 1));
    WizardForm.StatusLabel.Caption := Copy(Rest, 1, P - 1);
    WizardForm.FilenameLabel.Caption := Copy(Rest, P + 1, MaxInt);
  end else if Copy(S, 1, 6) = '@@pid ' then
    PyPid := StrToIntDef(Copy(S, 7, MaxInt), 0)
  else if Copy(S, 1, 8) = '@@error ' then
    ErrorText := Copy(S, 9, MaxInt)
  else if Copy(S, 1, 9) = '@@result ' then
    Outcome := Copy(S, 10, MaxInt)
  else if Copy(S, 1, 7) = '@@size ' then
    InstalledKB := StrToIntDef(Copy(S, 8, MaxInt), 0)
  else begin
    Log(S);
    AddDetail(S);
  end;
end;

function ModelArgs: String;
begin
  Result := '';
  if HasSwitch('/CPU') then
    Result := ' --cpu';
  if ModelPage.SelectedValueIndex = 1 then
    Result := Result + ' --small-only'
  else if not HasNvidia then
    Result := Result + ' --model large';  { (with an NVIDIA graphics card both is the default) }
end;

{ Cancel while install_ai_select.py runs (Setup's own Cancel does nothing once its files are
  copied): stop it, with everything it started }
procedure StopClick(Sender: TObject);
var
  Code: Integer;
begin
  if Stopping or (MsgBox('Stop the installation?' + #13#10#13#10 + 'What was downloaded so far is '
      + 'kept: run Setup again to continue.', mbConfirmation, MB_YESNO) <> IDYES) then
    exit;
  Stopping := True;
  WizardForm.StatusLabel.Caption := 'Stopping';
  if PyPid <> 0 then
    Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /T /PID ' + IntToStr(PyPid), '', SW_HIDE,
      ewWaitUntilTerminated, Code);
end;

procedure RunInstall;
var
  Code: Integer;
  DefaultCancel: TNotifyEvent;
  Params: String;
begin
  Running := True;
  DefaultCancel := WizardForm.CancelButton.OnClick;
  WizardForm.CancelButton.OnClick := @StopClick;
  WizardForm.CancelButton.Enabled := True;
  Base := 0;
  if PythonExe = '' then begin
    if not InstallPython then begin
      Running := False;
      WizardForm.CancelButton.OnClick := DefaultCancel;
      exit;
    end;
    Base := 40;
  end;
  SetIniString('setup', 'python', PythonExe, ExpandConstant('{app}\setup.ini'));
  if PythonBySetup then begin  { (for the uninstaller; kept when a later setup finds this Python) }
    SetIniString('setup', 'python_by_setup', '{#PyVersion}', ExpandConstant('{app}\setup.ini'));
    SetIniBool('setup', 'launcher_by_setup', not LauncherBefore, ExpandConstant('{app}\setup.ini'));
  end;
  WizardForm.StatusLabel.Caption := 'Getting ready';
  Params := '-u "' + ExpandConstant('{app}\install_ai_select.py') + '" --progress' + ModelArgs;
  Log('Running ' + PythonExe + ' ' + Params);
  try
    ExecAndLogOutput(PythonExe, Params, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code,
      @OnInstallerLine);
  except
    ErrorText := GetExceptionMessage;
  end;
  Running := False;
  WizardForm.CancelButton.OnClick := DefaultCancel;
end;

{ --- Finishing ------------------------------------------------------------------------ }

procedure ShowLogClick(Sender: TObject);
var
  Code: Integer;
begin
  ShellExec('open', ExpandConstant('{log}'), '', '', SW_SHOWNORMAL, ewNoWait, Code);
end;

procedure ShowOutcome;
var
  Button: TNewButton;
  Text: String;
begin
  if Outcome = 'passed' then
    exit;
  if Stopping then begin
    WizardForm.FinishedHeadingLabel.Caption := 'Setup was stopped';
    Text := 'What was downloaded so far is kept: run Setup again to continue.';
  end else if Outcome = 'failed' then begin
    WizardForm.FinishedHeadingLabel.Caption := 'Installed, but some tests failed';
    Text := 'AI Fuzzy Select is installed, but not everything worked in its tests. The log says '
      + 'what failed.';
  end else begin
    WizardForm.FinishedHeadingLabel.Caption := 'The installation failed';
    if ErrorText = '' then
      ErrorText := 'The installer stopped unexpectedly.';
    Text := ErrorText + #13#10#13#10 + 'Run Setup again to try again: what was downloaded is kept.';
  end;
  WizardForm.FinishedLabel.Caption := Text;
  WizardForm.AdjustLabelHeight(WizardForm.FinishedLabel);
  Button := TNewButton.Create(WizardForm);
  Button.Parent := WizardForm.FinishedPage;
  Button.Caption := 'Show the log';
  Button.SetBounds(WizardForm.FinishedLabel.Left, WizardForm.FinishedLabel.Top
    + WizardForm.FinishedLabel.Height + ScaleY(16), ScaleX(120), WizardForm.NextButton.Height);
  Button.OnClick := @ShowLogClick;
end;

function GimpExe(Param: String): String;
begin
  Result := GimpExePath;
end;

function CanStartGimp: Boolean;
begin
  Result := (GimpExePath <> '') and (Outcome = 'passed');
end;

{ --- Event functions ------------------------------------------------------------------ }

procedure InitializeWizard;
begin
  ExtractTemporaryFile('ok.png');
  ExtractTemporaryFile('missing.png');
  ExtractTemporaryFile('install.png');
  ExtractTemporaryFile('info.png');
  ExtractTemporaryFile('warn.png');
  CreateRequirementsPage;
  CreateModelPage;
  CreateInstallingControls;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = ReqPage.ID then
    CheckRequirements
  else if CurPageID = wpFinished then
    ShowOutcome;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = ReqPage.ID then
    Result := ReadyToGo
  else if CurPageID = ModelPage.ID then
    ModelChosen := True;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Estimate;
  Result := 'Setup will:' + NewLine;
  if PythonExe = '' then
    Result := Result + Space + 'install Python {#PyVersion} for your user,' + NewLine;
  if DownloadMB > 0 then
    Result := Result + Space + 'download the AI and its models: ' + SizeText(DownloadMB) + ',' + NewLine;
  Result := Result + Space + 'install AI Fuzzy Select in ' + GimpName + ' and test it.' + NewLine + NewLine;
  if ModelPage.SelectedValueIndex = 0 then
    Result := Result + 'AI models: SAM 2.1 large and small'
  else
    Result := Result + 'AI model: SAM 2.1 small';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Key: String;
begin
  if CurStep = ssPostInstall then
    RunInstall
  else if (CurStep = ssDone) and (InstalledKB > 0) then begin
    { the size under Settings > Apps: the download, not just the few files of this setup }
    Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{{#AppId}}_is1';
    RegWriteDWordValue(HKCU, Key, 'EstimatedSize', InstalledKB);
  end;
end;

{ --- Uninstalling --------------------------------------------------------------------- }

procedure RemoveWithoutPython;
var
  Find: TFindRec;
  Root: String;
begin
  DelTree(DataDir(''), True, True, True);
  Root := ExpandConstant('{userappdata}\GIMP\');
  if FindFirst(Root + '3.*', Find) then
    try
      repeat
        DelTree(Root + Find.Name + '\plug-ins\aiselect', True, True, True);
      until not FindNext(Find);
    finally
      FindClose(Find);
    end;
end;

{ Python, if this setup installed it, with the launcher if that came with it; and its folder,
  where uninstalling Python leaves files behind }
procedure RemovePython;
var
  Ini, Version, Python, Key, Cmd, Exe, Params: String;
  Code: Integer;
begin
  Ini := ExpandConstant('{app}\setup.ini');
  Version := GetIniString('setup', 'python_by_setup', '', Ini);
  if Version = '' then
    exit;
  Python := GetIniString('setup', 'python', '', Ini);
  Key := FindUninstallKey('Python ' + Version + ' ', True);
  if (Key = '') or not RegQueryStringValue(HKCU, UninstallKey + '\' + Key, 'UninstallString', Cmd) then
    exit;
  // like "C:\...\Package Cache\(id)\python-3.14.10-amd64.exe" /uninstall
  Cmd := Trim(Cmd);
  if Copy(Cmd, 1, 1) = '"' then begin
    Exe := Copy(Cmd, 2, Pos('"', Copy(Cmd, 2, MaxInt)) - 1);
    Params := Trim(Copy(Cmd, Length(Exe) + 3, MaxInt));
  end else begin
    Exe := Copy(Cmd, 1, Pos(' ', Cmd + ' ') - 1);
    Params := Trim(Copy(Cmd, Length(Exe) + 1, MaxInt));
  end;
  if Pos('/quiet', Lowercase(Params)) = 0 then
    Params := Params + ' /quiet';
  Log('Removing Python ' + Version + ': ' + Exe + ' ' + Params);
  if not Exec(Exe, Params, '', SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then begin
    Log(Format('The Python uninstaller failed (%d)', [Code]));
    exit;
  end;
  if GetIniBool('setup', 'launcher_by_setup', False, Ini) then begin
    Key := FindUninstallKey('Python Launcher', False);
    if Key <> '' then
      Exec(ExpandConstant('{sys}\msiexec.exe'), '/x ' + Key + ' /quiet /norestart', '', SW_HIDE,
        ewWaitUntilTerminated, Code);
  end;
  if (Python <> '') and DirExists(ExtractFileDir(Python)) then
    DelTree(ExtractFileDir(Python), True, True, True);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Python: String;
  Code: Integer;
begin
  if CurUninstallStep <> usUninstall then
    exit;
  { The plug-in, its virtual environment and downloads, by uninstall_ai_select.py: it also
    stops the plug-in's backend if it's running }
  Python := GetIniString('setup', 'python', '', ExpandConstant('{app}\setup.ini'));
  if FileExists(Python) then begin
    if not Exec(Python, '"' + ExpandConstant('{app}\uninstall_ai_select.py') + '" --yes', '', SW_HIDE,
        ewWaitUntilTerminated, Code) then
      RemoveWithoutPython
    else if (Code <> 0) and not UninstallSilent then
      MsgBox('Some files could not be removed, probably because GIMP is open. Close GIMP and '
        + 'delete the folder ' + DataDir('') + ' yourself.', mbInformation, MB_OK);
  end else
    RemoveWithoutPython;
  RemovePython;
end;
