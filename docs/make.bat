@ECHO OFF

REM Windows build script for Sphinx documentation.
REM Run from the docs\ directory.

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "html" goto html
if "%1" == "clean" goto clean
goto help

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html
echo.
echo Build finished. Open %BUILDDIR%\html\index.html to view.
goto end

:clean
rmdir /s /q %BUILDDIR%
goto end

:end

