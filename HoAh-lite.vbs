' Double-click to start HoAh-lite (no console window).
' First run creates the Python environment automatically via setup.bat.
Option Explicit
Dim shell, fso, root, pythonw, args, i
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pythonw) Then
    shell.Run "cmd /c """ & root & "\setup.bat""", 1, True
    If Not fso.FileExists(pythonw) Then
        MsgBox "Environment setup failed. Run setup.bat to see the error.", 16, "HoAh-lite"
        WScript.Quit 1
    End If
End If

args = ""
For i = 0 To WScript.Arguments.Count - 1
    args = args & " """ & WScript.Arguments(i) & """"
Next

shell.CurrentDirectory = root
shell.Run """" & pythonw & """ -m hoah_lite" & args, 0, False
