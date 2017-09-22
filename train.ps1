For ($k=4; $k -le 100; ) {
    For ($i=0; $i -le 4; $i++) {
        python train_nets.py --kappa=$k --id=$i --model=LAMP --T=6
        python train_nets.py --kappa=$k --id=$i --model=LVAMP --T=6
        python train_nets.py --kappa=$k --id=$i --model=LISTA --T=6
    }
    If ($k -eq 4) { $k = 5 }
    Else { $k = $k + 5 }
}
