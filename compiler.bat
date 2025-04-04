:: compiler for Program.cs
@echo OFF
dotnet publish --configuration Release --runtime win-x64 --self-contained true -p:PublishSingleFile=true -p:Version=0.0.2
