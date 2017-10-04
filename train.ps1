For ($k=4; $k -le 50; ) {
    python train_nets.py --kappa=$k --id=0 --model=LAMP --T=6
    python train_nets.py --kappa=$k --id=0 --model=LVAMP --T=6
    python train_nets.py --kappa=$k --id=0 --model=LISTA --T=6
    If ($k -eq 4) { $k = 5 }
    Else { $k = $k + 5 }
}
