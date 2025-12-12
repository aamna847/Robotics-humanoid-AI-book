@echo off
setlocal enabledelayedexpansion

REM Simple script to create a new ADR file with incremented ID
REM Usage: create-adc.bat "Title of the Architecture Decision"

if "%~1"=="" (
    echo Error: Title is required
    echo Usage: create-adr.bat "Title of the Architecture Decision"
    exit /b 1
)

REM Get the next ADR ID by looking at existing files
set "nextId=0"
for /f "tokens=*" %%a in ('dir /b "C:\humanoid-robotics\history\adr\*.md" 2^>nul') do (
    set "fileName=%%a"
    set "checkId=!fileName:~4,3!"
    for /f "tokens=*" %%b in ("!checkId!") do (
        if %%b gtr !nextId! set "nextId=%%b"
    )
)
set /a nextId+=1
if !nextId! LSS 10 set "nextId=00!nextId!"
if !nextId! LSS 100 set "nextId=0!nextId!"

REM Create the ADR file
set "adrFile=C:\humanoid-robotics\history\adr\ADR-!nextId!.md"
echo # ADR-!nextId!: %~1 > "!adrFile!"
echo. >> "!adrFile!"
echo Date: %date% >> "!adrFile!"
echo. >> "!adrFile!"
echo ## Status >> "!adrFile!"
echo Proposed >> "!adrFile!"
echo. >> "!adrFile!"
echo ## Context >> "!adrFile!"
echo Provide context for the decision here. >> "!adrFile!"
echo. >> "!adrFile!"
echo ## Decision >> "!adrFile!"
echo The decision that was made. >> "!adrFile!"
echo. >> "!adrFile!"
echo ## Consequences >> "!adrFile!"
echo The outcomes, tradeoffs, and risks of the decision. >> "!adrFile!"
echo. >> "!adrFile!"
echo ## Alternatives >> "!adrFile!"
echo Alternative approaches that were considered. >> "!adrFile!"
echo. >> "!adrFile!"
echo ## References >> "!adrFile!"
echo Links to related documents. >> "!adrFile!"

REM Output JSON response
echo {"adr_id":"!nextId!", "adr_path":"!adrFile!"}