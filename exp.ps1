For ($k=4; $k -le 40; ) {
    python exp.py $k 0
    If ($k -eq 4) { $k = 5 }
    Else { $k = $k + 5 }
}