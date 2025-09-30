[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20810464&assignment_repo_type=AssignmentRepo)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py


# Tensor Training

Ниже результаты обучения сети на разных наборах данных:

## Simple
**Config:** `PTS=50, HIDDEN=2, RATE=0.5`  

**Logs (excerpt):**

Epoch 10  loss 33.084963825821255  correct 30  time 44.9 ms
Epoch 20  loss 32.07883414281687  correct 30  time 43.9 ms
Epoch 30  loss 29.128612872936035  correct 41  time 47.8 ms
Epoch 40  loss 21.16425993767851  correct 50  time 43.9 ms
Epoch 50  loss 12.56690834815013  correct 49  time 42.9 ms
Epoch 60  loss 8.589036734404605  correct 50  time 42.8 ms
Epoch 70  loss 6.40475678784042  correct 50  time 49.4 ms
Epoch 80  loss 5.070986470290297  correct 50  time 43.0 ms
Epoch 90  loss 4.194961393958395  correct 50  time 43.1 ms
Epoch 100  loss 3.5719986906083623  correct 50  time 43.3 ms
Epoch 110  loss 3.105238518846003  correct 50  time 43.4 ms
Epoch 120  loss 2.746627557516956  correct 50  time 43.5 ms
Epoch 130  loss 2.4636243604541064  correct 50  time 43.2 ms
Epoch 140  loss 2.232205708805474  correct 50  time 43.8 ms
Epoch 150  loss 2.037578591944483  correct 50  time 44.1 ms
Epoch 160  loss 1.8734225445922617  correct 50  time 43.3 ms
Epoch 170  loss 1.7316972449970511  correct 50  time 42.9 ms
Epoch 180  loss 1.6060332038589389  correct 50  time 43.9 ms
Epoch 190  loss 1.4949708798676558  correct 50  time 49.8 ms
Epoch 200  loss 1.3961431892732796  correct 50  time 44.3 ms
Epoch 210  loss 1.3076873808908676  correct 50  time 47.9 ms
Epoch 220  loss 1.22688856348672  correct 50  time 44.1 ms
Epoch 230  loss 1.155196300244038  correct 50  time 44.4 ms
Epoch 240  loss 1.0901593052791576  correct 50  time 44.0 ms
Epoch 250  loss 1.0309593183017474  correct 50  time 43.6 ms
Epoch 260  loss 0.97690639317174  correct 50  time 43.9 ms
Epoch 270  loss 0.9274132493941805  correct 50  time 44.1 ms
Epoch 280  loss 0.8819760237694386  correct 50  time 43.7 ms
Epoch 290  loss 0.8405493131293078  correct 50  time 44.0 ms
Epoch 300  loss 0.8017658262350755  correct 50  time 44.3 ms
Epoch 310  loss 0.7661638403165473  correct 50  time 44.2 ms
Epoch 320  loss 0.7332199322367127  correct 50  time 44.1 ms
Epoch 330  loss 0.7026677997914307  correct 50  time 44.5 ms
Epoch 340  loss 0.6742733683654506  correct 50  time 50.9 ms
Epoch 350  loss 0.6478309948582669  correct 50  time 45.5 ms
Epoch 360  loss 0.6231593490108729  correct 50  time 46.6 ms
Epoch 370  loss 0.6000979898705523  correct 50  time 44.4 ms
Epoch 380  loss 0.5785045029458681  correct 50  time 43.8 ms
Epoch 390  loss 0.5580900627407577  correct 50  time 44.3 ms
Epoch 400  loss 0.5390772129761159  correct 50  time 44.3 ms
Epoch 410  loss 0.5212037829129009  correct 50  time 43.7 ms
Epoch 420  loss 0.5043744892411008  correct 50  time 43.1 ms
Epoch 430  loss 0.48850603441333784  correct 50  time 43.6 ms
Epoch 440  loss 0.4735207084486569  correct 50  time 45.7 ms
Epoch 450  loss 0.45934912519090215  correct 50  time 43.3 ms
Epoch 460  loss 0.44592907129518  correct 50  time 43.6 ms
Epoch 470  loss 0.4332045595893138  correct 50  time 51.3 ms
Epoch 480  loss 0.42112503634691467  correct 50  time 49.7 ms
Epoch 490  loss 0.40964471331674246  correct 50  time 43.0 ms
Epoch 500  loss 0.39872200138665126  correct 50  time 42.8 ms 


## Diag
**Config:** `PTS=50, HIDDEN=3, RATE=0.5, EPOCHS=800`  

**Logs:**

Epoch 10  loss 18.601860814369505  correct 41  time 67.5 ms
Epoch 20  loss 14.314058265608995  correct 41  time 68.6 ms
Epoch 30  loss 11.040774381156666  correct 41  time 66.5 ms
Epoch 40  loss 8.966614728766888  correct 46  time 74.1 ms
Epoch 50  loss 7.459373068910147  correct 49  time 67.7 ms
Epoch 60  loss 6.388054029158053  correct 50  time 77.5 ms
Epoch 70  loss 5.559587622071403  correct 50  time 68.9 ms
Epoch 80  loss 4.902267125782875  correct 50  time 68.3 ms
Epoch 90  loss 4.355946089765352  correct 50  time 68.6 ms
Epoch 100  loss 3.9084674140670512  correct 50  time 74.5 ms
Epoch 110  loss 3.5400079535804023  correct 50  time 68.1 ms
Epoch 120  loss 3.2288115947597094  correct 50  time 67.8 ms
Epoch 130  loss 2.896314452721042  correct 50  time 69.1 ms
Epoch 140  loss 2.675604183964001  correct 50  time 73.5 ms
Epoch 150  loss 2.451260567433503  correct 50  time 68.1 ms
Epoch 160  loss 2.286067029643192  correct 50  time 78.8 ms
Epoch 170  loss 2.1440557025097564  correct 50  time 68.2 ms
Epoch 180  loss 2.016161531471337  correct 50  time 66.3 ms
Epoch 190  loss 1.9016323488924554  correct 50  time 68.2 ms
Epoch 200  loss 1.798294152208496  correct 50  time 66.0 ms
Epoch 210  loss 1.704450123641912  correct 50  time 72.4 ms
Epoch 220  loss 1.6187576025996147  correct 50  time 76.1 ms
Epoch 230  loss 1.5401390423680745  correct 50  time 71.8 ms
Epoch 240  loss 1.4677182054879618  correct 50  time 67.7 ms
Epoch 250  loss 1.3955319601169753  correct 50  time 68.0 ms
Epoch 260  loss 1.3347473795951557  correct 50  time 65.7 ms
Epoch 270  loss 1.2782790740224699  correct 50  time 66.7 ms
Epoch 280  loss 1.225669033824838  correct 50  time 66.4 ms
Epoch 290  loss 1.1765289061554516  correct 50  time 67.7 ms
Epoch 300  loss 1.130526226674445  correct 50  time 66.9 ms
Epoch 310  loss 1.087373841143939  correct 50  time 67.7 ms
Epoch 320  loss 1.0468217042124162  correct 50  time 66.7 ms
Epoch 330  loss 1.0086504640064649  correct 50  time 66.3 ms
Epoch 340  loss 0.9726663984796264  correct 50  time 66.0 ms
Epoch 350  loss 0.9386973825912172  correct 50  time 66.7 ms
Epoch 360  loss 0.9065896474182322  correct 50  time 75.9 ms
Epoch 370  loss 0.8762051522345037  correct 50  time 77.9 ms
Epoch 380  loss 0.8474194346592153  correct 50  time 66.4 ms
Epoch 390  loss 0.8201198365646134  correct 50  time 67.6 ms
Epoch 400  loss 0.7942040276696142  correct 50  time 68.4 ms
Epoch 410  loss 0.7695787668472893  correct 50  time 67.7 ms
Epoch 420  loss 0.7461588547851373  correct 50  time 68.8 ms
Epoch 430  loss 0.7238662419096621  correct 50  time 68.2 ms
Epoch 440  loss 0.7026292632886727  correct 50  time 66.3 ms
Epoch 450  loss 0.682381978183033  correct 50  time 68.0 ms
Epoch 460  loss 0.663063596496125  correct 50  time 67.9 ms
Epoch 470  loss 0.6446179779061575  correct 50  time 67.3 ms
Epoch 480  loss 0.6269931922166743  correct 50  time 66.4 ms
Epoch 490  loss 0.6101411316127547  correct 50  time 67.7 ms
Epoch 500  loss 0.5940171672054713  correct 50  time 68.1 ms
Epoch 510  loss 0.5785798435906994  correct 50  time 67.6 ms
Epoch 520  loss 0.5637906062203737  correct 50  time 67.5 ms
Epoch 530  loss 0.5496135572449926  correct 50  time 69.1 ms
Epoch 540  loss 0.5360152361818594  correct 50  time 75.2 ms
Epoch 550  loss 0.522964422329177  correct 50  time 65.9 ms
Epoch 560  loss 0.510431956309153  correct 50  time 69.8 ms
Epoch 570  loss 0.4983905785043379  correct 50  time 66.4 ms
Epoch 580  loss 0.4868147824671829  correct 50  time 70.4 ms
Epoch 590  loss 0.47568068164574306  correct 50  time 66.6 ms
Epoch 600  loss 0.46496588798881966  correct 50  time 68.8 ms
Epoch 610  loss 0.4546494011793823  correct 50  time 67.9 ms
Epoch 620  loss 0.4447115074022907  correct 50  time 67.7 ms
Epoch 630  loss 0.4351336866860075  correct 50  time 74.4 ms
Epoch 640  loss 0.4258985279724457  correct 50  time 66.8 ms
Epoch 650  loss 0.4169896511671979  correct 50  time 75.6 ms
Epoch 660  loss 0.40839163550731294  correct 50  time 66.7 ms
Epoch 670  loss 0.400089953657095  correct 50  time 68.1 ms
Epoch 680  loss 0.39207091100636104  correct 50  time 67.8 ms
Epoch 690  loss 0.3843215897013417  correct 50  time 67.6 ms
Epoch 700  loss 0.37682979698726726  correct 50  time 67.6 ms
Epoch 710  loss 0.36958401748466235  correct 50  time 66.3 ms
Epoch 720  loss 0.3625733690592269  correct 50  time 67.8 ms
Epoch 730  loss 0.3557875619786957  correct 50  time 68.2 ms
Epoch 740  loss 0.3492168610797482  correct 50  time 68.0 ms
Epoch 750  loss 0.3428520506944632  correct 50  time 67.0 ms
Epoch 760  loss 0.3366844021093617  correct 50  time 67.6 ms
Epoch 770  loss 0.3307056433510603  correct 50  time 67.9 ms
Epoch 780  loss 0.3249079311113979  correct 50  time 66.9 ms
Epoch 790  loss 0.3192838246417647  correct 50  time 73.2 ms
Epoch 800  loss 0.31382626146152937  correct 50  time 66.5 ms


## Split
**Config:** `PTS=50, HIDDEN=6, RATE=0.5, EPOCHS=800`  

**Logs:**

Epoch 10  loss 31.824720864138083  correct 32  time 174.7 ms
Epoch 20  loss 31.225558953737295  correct 32  time 167.0 ms
Epoch 30  loss 29.990453985232023  correct 32  time 185.1 ms
Epoch 40  loss 27.76115840961988  correct 40  time 186.5 ms
Epoch 50  loss 25.233854995155482  correct 43  time 173.1 ms
Epoch 60  loss 23.218501691530356  correct 43  time 173.1 ms
Epoch 70  loss 28.953638964960323  correct 33  time 169.5 ms
Epoch 80  loss 25.677154018004867  correct 35  time 171.1 ms
Epoch 90  loss 25.034470692990524  correct 36  time 193.5 ms
Epoch 100  loss 24.270828400981046  correct 37  time 231.6 ms
Epoch 110  loss 23.706943252998133  correct 37  time 172.7 ms
Epoch 120  loss 23.18485072949778  correct 38  time 174.8 ms
Epoch 130  loss 22.52726109993301  correct 38  time 174.6 ms
Epoch 140  loss 21.91100736909124  correct 39  time 173.0 ms
Epoch 150  loss 21.312850461044942  correct 39  time 172.4 ms
Epoch 160  loss 20.11639518781999  correct 39  time 168.8 ms
Epoch 170  loss 20.16635703530091  correct 39  time 171.0 ms
Epoch 180  loss 19.0238197186029  correct 39  time 168.1 ms
Epoch 190  loss 18.740201869735344  correct 39  time 170.7 ms
Epoch 200  loss 17.731730653579923  correct 40  time 168.2 ms
Epoch 210  loss 17.143583982318454  correct 40  time 198.4 ms
Epoch 220  loss 16.59622752865805  correct 40  time 171.5 ms
Epoch 230  loss 16.00125058319847  correct 41  time 171.8 ms
Epoch 240  loss 15.471800570679276  correct 41  time 170.8 ms
Epoch 250  loss 15.081560205507596  correct 41  time 171.2 ms
Epoch 260  loss 14.46752569336596  correct 42  time 169.4 ms
Epoch 270  loss 13.591323428164506  correct 42  time 173.6 ms
Epoch 280  loss 13.735901688631829  correct 42  time 193.1 ms
Epoch 290  loss 12.737268409653653  correct 42  time 235.7 ms
Epoch 300  loss 12.43041789648088  correct 42  time 173.7 ms
Epoch 310  loss 12.288093596894024  correct 42  time 174.1 ms
Epoch 320  loss 11.823586389273482  correct 42  time 172.7 ms
Epoch 330  loss 11.584397403946507  correct 42  time 173.3 ms
Epoch 340  loss 11.333584883631946  correct 42  time 195.3 ms
Epoch 350  loss 10.9190888616571  correct 43  time 174.1 ms
Epoch 360  loss 10.564958981198245  correct 43  time 174.5 ms
Epoch 370  loss 10.576855949069422  correct 44  time 183.4 ms
Epoch 380  loss 10.125003041203033  correct 44  time 193.6 ms
Epoch 390  loss 9.957535373080612  correct 45  time 174.8 ms
Epoch 400  loss 9.755518308020081  correct 45  time 173.4 ms
Epoch 410  loss 9.470025286942956  correct 45  time 172.0 ms
Epoch 420  loss 9.161735415542335  correct 45  time 173.0 ms
Epoch 430  loss 9.073258025593269  correct 45  time 194.6 ms
Epoch 440  loss 8.853281454709848  correct 45  time 173.2 ms
Epoch 450  loss 8.54984543312736  correct 45  time 173.9 ms
Epoch 460  loss 8.549146647994306  correct 45  time 175.3 ms
Epoch 470  loss 8.232511880571277  correct 45  time 173.3 ms
Epoch 480  loss 8.219437544392555  correct 45  time 173.7 ms
Epoch 490  loss 8.09623967551599  correct 45  time 174.2 ms
Epoch 500  loss 7.912267648503052  correct 45  time 175.1 ms
Epoch 510  loss 7.5727263609204325  correct 45  time 169.1 ms
Epoch 520  loss 7.356930758688622  correct 46  time 174.8 ms
Epoch 530  loss 7.341880827626502  correct 46  time 174.6 ms
Epoch 540  loss 7.197241747899883  correct 46  time 183.2 ms
Epoch 550  loss 7.061272761839571  correct 47  time 174.2 ms
Epoch 560  loss 6.738076900646351  correct 47  time 203.0 ms
Epoch 570  loss 6.339204146618526  correct 47  time 170.2 ms
Epoch 580  loss 6.574897303843815  correct 47  time 171.9 ms
Epoch 590  loss 7.574574485157803  correct 45  time 172.4 ms
Epoch 600  loss 6.304591893793114  correct 47  time 171.5 ms
Epoch 610  loss 4.370887195888381  correct 48  time 174.8 ms
Epoch 620  loss 3.9584461391641845  correct 48  time 173.3 ms
Epoch 630  loss 5.541240033368986  correct 48  time 173.1 ms
Epoch 640  loss 9.179262378433837  correct 45  time 170.5 ms
Epoch 650  loss 4.809768315184137  correct 48  time 174.0 ms
Epoch 660  loss 3.092140503791101  correct 49  time 172.4 ms
Epoch 670  loss 2.9536461835369923  correct 49  time 172.7 ms
Epoch 680  loss 3.612681956607102  correct 49  time 169.7 ms
Epoch 690  loss 11.002379066442034  correct 44  time 188.1 ms
Epoch 700  loss 3.8835478739790714  correct 49  time 174.1 ms
Epoch 710  loss 2.7585894545115774  correct 50  time 197.0 ms
Epoch 720  loss 2.641298447222444  correct 49  time 169.9 ms
Epoch 730  loss 3.0024801147495257  correct 49  time 172.5 ms
Epoch 740  loss 6.547168469488265  correct 47  time 175.0 ms
Epoch 750  loss 7.911861525032584  correct 45  time 179.3 ms
Epoch 760  loss 2.7142511355669954  correct 49  time 198.5 ms
Epoch 770  loss 2.8797849503137183  correct 49  time 169.3 ms
Epoch 780  loss 3.4226573815103145  correct 49  time 170.7 ms
Epoch 790  loss 3.2181351987546085  correct 49  time 175.3 ms
Epoch 800  loss 3.208793849095823  correct 49  time 183.1 ms

## XOR
**Config:** `PTS=50, HIDDEN=10, RATE=0.5, EPOCHS=800`  

**Logs:**

Epoch 10  loss 26.922552148730794  correct 39  time 379.8 ms
Epoch 20  loss 27.881670327414923  correct 32  time 389.8 ms
Epoch 30  loss 20.66052586253231  correct 43  time 410.8 ms
Epoch 40  loss 20.355053806470117  correct 42  time 382.3 ms
Epoch 50  loss 18.65098936589013  correct 42  time 377.6 ms
Epoch 60  loss 16.16053181385229  correct 46  time 396.5 ms
Epoch 70  loss 15.591174075985075  correct 45  time 399.7 ms
Epoch 80  loss 16.00097178050949  correct 45  time 381.8 ms
Epoch 90  loss 17.896117768242075  correct 44  time 383.0 ms
Epoch 100  loss 15.354289586497174  correct 43  time 420.7 ms
Epoch 110  loss 14.862492273496315  correct 43  time 411.0 ms
Epoch 120  loss 17.505711191721044  correct 42  time 383.2 ms
Epoch 130  loss 14.134323278571898  correct 43  time 381.4 ms
Epoch 140  loss 13.882692708036052  correct 43  time 381.8 ms
Epoch 150  loss 14.424757299449873  correct 43  time 386.5 ms
Epoch 160  loss 15.056360719747858  correct 41  time 432.5 ms
Epoch 170  loss 15.324021041955957  correct 42  time 375.7 ms
Epoch 180  loss 14.33673451279534  correct 42  time 382.2 ms
Epoch 190  loss 14.065925066136474  correct 42  time 418.5 ms
Epoch 200  loss 14.143781478293931  correct 42  time 378.0 ms
Epoch 210  loss 14.187746907198242  correct 43  time 379.7 ms
Epoch 220  loss 14.074607339162252  correct 43  time 391.1 ms
Epoch 230  loss 13.855933259743734  correct 43  time 380.6 ms
Epoch 240  loss 13.941733362400756  correct 42  time 427.1 ms
Epoch 250  loss 14.57366610994968  correct 43  time 394.8 ms
Epoch 260  loss 15.264805112586318  correct 42  time 406.1 ms
Epoch 270  loss 14.159746216506147  correct 44  time 375.5 ms
Epoch 280  loss 14.05678116147734  correct 44  time 379.6 ms
Epoch 290  loss 13.843011352887324  correct 44  time 373.6 ms
Epoch 300  loss 13.779499748736127  correct 44  time 375.1 ms
Epoch 310  loss 13.735008503540577  correct 44  time 420.5 ms
Epoch 320  loss 13.698795072140005  correct 44  time 384.4 ms
Epoch 330  loss 13.63252954175233  correct 44  time 412.8 ms
Epoch 340  loss 13.542367106596053  correct 44  time 390.3 ms
Epoch 350  loss 13.54347002019136  correct 44  time 412.3 ms
Epoch 360  loss 13.506850927451772  correct 44  time 407.6 ms
Epoch 370  loss 13.482085186766419  correct 44  time 385.5 ms
Epoch 380  loss 13.488798897491693  correct 44  time 387.0 ms
Epoch 390  loss 13.472394234954361  correct 44  time 378.7 ms
Epoch 400  loss 13.436996259382369  correct 44  time 386.9 ms
Epoch 410  loss 13.021691371716267  correct 44  time 379.7 ms
Epoch 420  loss 12.800202119895538  correct 46  time 378.3 ms
Epoch 430  loss 13.178118773321852  correct 45  time 379.7 ms
Epoch 440  loss 13.010500253055298  correct 45  time 403.0 ms
Epoch 450  loss 12.717614040356558  correct 44  time 376.4 ms
Epoch 460  loss 12.567404699484532  correct 44  time 417.6 ms
Epoch 470  loss 12.478771802087575  correct 44  time 384.9 ms
Epoch 480  loss 12.531750191703122  correct 43  time 385.7 ms
Epoch 490  loss 12.994234801231684  correct 44  time 386.0 ms
Epoch 500  loss 12.730519961857176  correct 44  time 383.2 ms
Epoch 510  loss 12.498529888116158  correct 43  time 424.5 ms
Epoch 520  loss 12.36070389669266  correct 44  time 381.7 ms
Epoch 530  loss 12.310136837915712  correct 43  time 379.0 ms
Epoch 540  loss 12.011238528468493  correct 44  time 386.7 ms
Epoch 550  loss 12.37771647942824  correct 43  time 377.4 ms
Epoch 560  loss 11.855089150889535  correct 44  time 412.4 ms
Epoch 570  loss 11.80850069314368  correct 45  time 379.1 ms
Epoch 580  loss 12.295413088124931  correct 44  time 380.6 ms
Epoch 590  loss 12.060195213936202  correct 44  time 375.0 ms
Epoch 600  loss 12.079459073828476  correct 44  time 372.5 ms
Epoch 610  loss 11.50202840463117  correct 45  time 378.4 ms
Epoch 620  loss 11.538463156440598  correct 45  time 373.1 ms
Epoch 630  loss 11.740439785004144  correct 43  time 374.8 ms
Epoch 640  loss 11.24989623213295  correct 46  time 380.6 ms
Epoch 650  loss 11.463921979114724  correct 45  time 379.0 ms
Epoch 660  loss 11.996893663048686  correct 43  time 378.4 ms
Epoch 670  loss 11.7406177534674  correct 44  time 376.0 ms
Epoch 680  loss 11.3632679778455  correct 45  time 373.9 ms
Epoch 690  loss 10.65705696638037  correct 46  time 378.3 ms
Epoch 700  loss 10.831695027739343  correct 46  time 380.3 ms
Epoch 710  loss 12.241097716902118  correct 42  time 382.8 ms
Epoch 720  loss 11.526573722516822  correct 44  time 382.5 ms
Epoch 730  loss 10.620222705379987  correct 46  time 396.2 ms
Epoch 740  loss 10.081127163609125  correct 46  time 377.3 ms
Epoch 750  loss 10.533004929699862  correct 45  time 380.0 ms
Epoch 760  loss 10.238225666613692  correct 45  time 417.9 ms
Epoch 770  loss 12.296109864926908  correct 44  time 374.8 ms
Epoch 780  loss 11.938492630772972  correct 46  time 388.7 ms
Epoch 790  loss 9.21977955936401  correct 46  time 380.4 ms
Epoch 800  loss 9.089013950932213  correct 46  time 383.0 ms


## Circle
**Config:** `PTS=50, HIDDEN=12, RATE=0.5, EPOCHS=1200`  

**Logs:**

Epoch 10  loss 29.584655337879045  correct 35  time 512.4 ms
Epoch 20  loss 29.103045661718  correct 35  time 510.4 ms
Epoch 30  loss 28.76682972777047  correct 35  time 526.7 ms
Epoch 40  loss 28.483727129393774  correct 35  time 516.6 ms
Epoch 50  loss 28.299986746887665  correct 35  time 506.7 ms
Epoch 60  loss 28.150863759373138  correct 35  time 509.8 ms
Epoch 70  loss 28.067093275736365  correct 34  time 509.4 ms
Epoch 80  loss 27.968610089731637  correct 32  time 515.0 ms
Epoch 90  loss 27.518379672780465  correct 32  time 516.4 ms
Epoch 100  loss 27.10434956057844  correct 34  time 510.3 ms
Epoch 110  loss 26.487431582477893  correct 36  time 513.8 ms
Epoch 120  loss 26.000748800202267  correct 35  time 509.2 ms
Epoch 130  loss 25.85440843837312  correct 35  time 572.8 ms
Epoch 140  loss 25.605767039506084  correct 37  time 508.6 ms
Epoch 150  loss 25.551258694192768  correct 38  time 514.5 ms
Epoch 160  loss 24.93174371224982  correct 38  time 514.4 ms
Epoch 170  loss 24.195578927493617  correct 39  time 523.6 ms
Epoch 180  loss 24.906125106525415  correct 39  time 573.4 ms
Epoch 190  loss 23.184457310002546  correct 39  time 535.1 ms
Epoch 200  loss 23.831022917214746  correct 39  time 513.6 ms
Epoch 210  loss 22.798643043190612  correct 39  time 574.2 ms
Epoch 220  loss 22.87339461388032  correct 38  time 511.1 ms
Epoch 230  loss 22.978648487463513  correct 38  time 507.9 ms
Epoch 240  loss 22.07284346251111  correct 38  time 520.9 ms
Epoch 250  loss 22.52313100390463  correct 40  time 513.4 ms
Epoch 260  loss 21.639500644978874  correct 40  time 559.8 ms
Epoch 270  loss 20.28767219047781  correct 41  time 516.1 ms
Epoch 280  loss 21.85815189818233  correct 38  time 511.8 ms
Epoch 290  loss 18.467791882028287  correct 42  time 578.1 ms
Epoch 300  loss 18.733493703322797  correct 44  time 514.5 ms
Epoch 310  loss 18.84344696658773  correct 41  time 508.5 ms
Epoch 320  loss 19.063315871988554  correct 41  time 510.2 ms
Epoch 330  loss 18.556924536049486  correct 42  time 510.7 ms
Epoch 340  loss 19.139057556735718  correct 42  time 513.3 ms
Epoch 350  loss 19.03301918271314  correct 41  time 518.8 ms
Epoch 360  loss 18.66779606789019  correct 41  time 514.8 ms
Epoch 370  loss 18.65914789967445  correct 41  time 518.3 ms
Epoch 380  loss 18.22740600704803  correct 42  time 509.7 ms
Epoch 390  loss 17.98092448388079  correct 42  time 511.2 ms
Epoch 400  loss 17.81671798653357  correct 41  time 576.0 ms
Epoch 410  loss 18.398159237377683  correct 41  time 512.0 ms
Epoch 420  loss 16.676610607114224  correct 42  time 511.2 ms
Epoch 430  loss 16.78396985918575  correct 42  time 507.5 ms
Epoch 440  loss 15.444945274809118  correct 43  time 564.0 ms
Epoch 450  loss 15.37819674013021  correct 44  time 518.3 ms
Epoch 460  loss 15.010575310074309  correct 44  time 513.8 ms
Epoch 470  loss 14.721282404318245  correct 44  time 509.9 ms
Epoch 480  loss 14.644247624566702  correct 44  time 565.3 ms
Epoch 490  loss 14.665776555369591  correct 44  time 507.5 ms
Epoch 500  loss 14.60994980887149  correct 43  time 506.9 ms
Epoch 510  loss 14.255053020376467  correct 43  time 513.6 ms
Epoch 520  loss 14.264922364233541  correct 43  time 563.7 ms
Epoch 530  loss 13.590882291839407  correct 43  time 515.1 ms
Epoch 540  loss 13.521430161096097  correct 43  time 507.8 ms
Epoch 550  loss 12.65164445448316  correct 43  time 507.4 ms
Epoch 560  loss 12.597343681986976  correct 43  time 558.1 ms
Epoch 570  loss 12.057777344913797  correct 43  time 509.9 ms
Epoch 580  loss 12.132570661502863  correct 43  time 508.6 ms
Epoch 590  loss 11.068610350826015  correct 43  time 514.5 ms
Epoch 600  loss 10.785382849515866  correct 44  time 541.2 ms
Epoch 610  loss 11.049227137578482  correct 43  time 510.9 ms
Epoch 620  loss 8.686387973676753  correct 48  time 509.9 ms
Epoch 630  loss 13.376173358202182  correct 42  time 506.5 ms
Epoch 640  loss 7.175595926930542  correct 48  time 566.5 ms
Epoch 650  loss 15.817131451472944  correct 42  time 571.4 ms
Epoch 660  loss 6.953690896298923  correct 48  time 574.6 ms
Epoch 670  loss 11.992027771455446  correct 42  time 509.4 ms
Epoch 680  loss 8.670692893910472  correct 47  time 513.2 ms
Epoch 690  loss 5.798303140068833  correct 50  time 512.8 ms
Epoch 700  loss 21.28481440308573  correct 39  time 511.2 ms
Epoch 710  loss 6.463728098779152  correct 48  time 513.3 ms
Epoch 720  loss 5.022845348820288  correct 50  time 512.6 ms
Epoch 730  loss 17.008479273329492  correct 42  time 506.8 ms
Epoch 740  loss 7.084806328731226  correct 48  time 510.5 ms
Epoch 750  loss 4.812527300566329  correct 50  time 515.5 ms
Epoch 760  loss 15.19898754006837  correct 41  time 511.8 ms
Epoch 770  loss 7.8773878190608935  correct 47  time 517.5 ms
Epoch 780  loss 4.539138576360946  correct 50  time 509.0 ms
Epoch 790  loss 7.293361089838174  correct 47  time 509.7 ms
Epoch 800  loss 12.809162963682397  correct 41  time 509.6 ms
Epoch 810  loss 4.933151808711325  correct 49  time 505.3 ms
Epoch 820  loss 4.018062862125074  correct 50  time 506.5 ms
Epoch 830  loss 7.708508718136584  correct 47  time 508.6 ms
Epoch 840  loss 11.428691647829272  correct 43  time 572.4 ms
Epoch 850  loss 4.167212910091316  correct 49  time 516.7 ms
Epoch 860  loss 3.8736253777834073  correct 49  time 508.6 ms
Epoch 870  loss 4.285387312946302  correct 47  time 510.0 ms
Epoch 880  loss 6.842538546290118  correct 47  time 520.7 ms
Epoch 890  loss 39.69951551179088  correct 40  time 509.8 ms
Epoch 900  loss 4.249684772135951  correct 49  time 515.4 ms
Epoch 910  loss 4.013434318185883  correct 49  time 509.8 ms
Epoch 920  loss 6.116582856937972  correct 48  time 509.9 ms
Epoch 930  loss 3.667293195912869  correct 49  time 534.2 ms
Epoch 940  loss 7.6162434133987675  correct 46  time 543.5 ms
Epoch 950  loss 3.3846034590830683  correct 50  time 572.9 ms
Epoch 960  loss 6.315229223189594  correct 48  time 569.2 ms
Epoch 970  loss 3.6997226402689956  correct 49  time 515.4 ms
Epoch 980  loss 7.093601057969069  correct 47  time 519.3 ms
Epoch 990  loss 31.030256936624237  correct 37  time 511.3 ms
Epoch 1000  loss 4.638490997763813  correct 49  time 513.5 ms
Epoch 1010  loss 4.07322996824757  correct 49  time 521.5 ms
Epoch 1020  loss 4.0123356511887796  correct 49  time 570.0 ms
Epoch 1030  loss 4.517450152762698  correct 48  time 509.7 ms
Epoch 1040  loss 4.057522091142845  correct 48  time 510.3 ms
Epoch 1050  loss 4.961740200800199  correct 47  time 511.6 ms
Epoch 1060  loss 3.4833497086982073  correct 49  time 535.8 ms
Epoch 1070  loss 6.571156523569295  correct 47  time 539.9 ms
Epoch 1080  loss 3.2468008230711836  correct 49  time 538.0 ms
Epoch 1090  loss 6.234826795250012  correct 47  time 520.2 ms
Epoch 1100  loss 33.74335819424019  correct 40  time 513.4 ms
Epoch 1110  loss 4.930328516286455  correct 48  time 515.6 ms
Epoch 1120  loss 4.174532024817745  correct 48  time 513.6 ms
Epoch 1130  loss 3.784116953620792  correct 48  time 574.9 ms
Epoch 1140  loss 3.2640780452582434  correct 48  time 511.7 ms
Epoch 1150  loss 3.0373376275272372  correct 48  time 507.6 ms
Epoch 1160  loss 2.739599796691807  correct 50  time 594.3 ms
Epoch 1170  loss 2.4268063187418782  correct 50  time 516.8 ms
Epoch 1180  loss 2.074840750252406  correct 50  time 563.4 ms
Epoch 1190  loss 1.9469458042854177  correct 50  time 509.8 ms
Epoch 1200  loss 1.9238387146787639  correct 50  time 519.8 ms


## Spiral
**Config:** `PTS=100, HIDDEN=20, RATE=0.5, EPOCHS=1500`  

**Logs:**

Epoch 10  loss 69.1921368067343  correct 52  time 2681.0 ms
Epoch 20  loss 68.32413157603402  correct 54  time 2572.0 ms
Epoch 30  loss 67.70844613490138  correct 57  time 2585.2 ms
Epoch 40  loss 67.186774549853  correct 55  time 2493.3 ms
Epoch 50  loss 66.68262609537439  correct 59  time 2736.5 ms
Epoch 60  loss 66.22211161444584  correct 59  time 2487.0 ms
Epoch 70  loss 65.91290216986535  correct 61  time 2495.7 ms
Epoch 80  loss 65.61255767798879  correct 60  time 2523.9 ms
Epoch 90  loss 65.26139061598995  correct 59  time 2513.3 ms
Epoch 100  loss 65.02133658735423  correct 59  time 2491.2 ms
Epoch 110  loss 64.95648568829814  correct 61  time 2631.0 ms
Epoch 120  loss 64.71103794110934  correct 59  time 2508.8 ms
Epoch 130  loss 64.47322664690684  correct 61  time 2766.8 ms
Epoch 140  loss 64.3034995896889  correct 60  time 2508.4 ms
Epoch 150  loss 64.16897896453303  correct 61  time 2510.0 ms
Epoch 160  loss 64.20654000075925  correct 61  time 2493.6 ms
Epoch 170  loss 64.06221466449624  correct 60  time 2492.1 ms
Epoch 180  loss 64.53433254855021  correct 58  time 2494.8 ms
Epoch 190  loss 64.13100967337107  correct 59  time 2488.8 ms
Epoch 200  loss 63.811160102074105  correct 58  time 2632.8 ms
Epoch 210  loss 63.828773152484324  correct 59  time 2517.3 ms
Epoch 220  loss 63.944313606481344  correct 59  time 2515.9 ms
Epoch 230  loss 63.59498220808092  correct 60  time 2484.7 ms
Epoch 240  loss 63.43311766535439  correct 61  time 2490.4 ms
Epoch 250  loss 63.45097705686714  correct 59  time 2491.0 ms
Epoch 260  loss 63.92289182699942  correct 58  time 2512.6 ms
Epoch 270  loss 63.2149667388105  correct 62  time 2513.8 ms
Epoch 280  loss 62.99260802669785  correct 61  time 2497.6 ms
Epoch 290  loss 63.988602516977366  correct 58  time 2500.2 ms
Epoch 300  loss 63.15230332556206  correct 60  time 2532.9 ms
Epoch 310  loss 62.81968292412049  correct 62  time 2683.1 ms
Epoch 320  loss 63.56455817011932  correct 61  time 2502.8 ms
Epoch 330  loss 62.92247780185918  correct 60  time 2513.7 ms
Epoch 340  loss 62.782426265856486  correct 61  time 2631.1 ms
Epoch 350  loss 63.25930757361364  correct 61  time 2496.4 ms
Epoch 360  loss 62.88966010517453  correct 62  time 2610.6 ms
Epoch 370  loss 62.410144860005715  correct 64  time 2748.4 ms
Epoch 380  loss 62.70739672528294  correct 63  time 2507.5 ms
Epoch 390  loss 62.72339397474323  correct 63  time 2503.8 ms
Epoch 400  loss 62.11646350097608  correct 64  time 2508.6 ms
Epoch 410  loss 62.17257803691634  correct 64  time 2493.3 ms
Epoch 420  loss 63.27041286830915  correct 61  time 2754.8 ms
Epoch 430  loss 61.965134734709274  correct 65  time 2486.4 ms
Epoch 440  loss 61.62468709161624  correct 64  time 2532.9 ms
Epoch 450  loss 62.45106626235832  correct 63  time 2479.5 ms
Epoch 460  loss 62.0455382111292  correct 63  time 2541.2 ms
Epoch 470  loss 61.61382710125821  correct 63  time 2483.9 ms
Epoch 480  loss 62.36247383718838  correct 65  time 2480.4 ms
Epoch 490  loss 61.845995425312665  correct 66  time 2688.6 ms
Epoch 500  loss 61.2337686873349  correct 64  time 2484.6 ms
Epoch 510  loss 62.18396298288098  correct 64  time 2579.7 ms
Epoch 520  loss 62.17751206997302  correct 65  time 2640.5 ms
Epoch 530  loss 60.93337680489822  correct 65  time 2506.3 ms
Epoch 540  loss 61.57199684614657  correct 65  time 2488.8 ms
Epoch 550  loss 61.57878599636234  correct 66  time 2532.9 ms
Epoch 560  loss 61.16997308012006  correct 67  time 2562.4 ms
Epoch 570  loss 61.06064360094894  correct 66  time 2496.1 ms
Epoch 580  loss 61.49825835254834  correct 65  time 2488.2 ms
Epoch 590  loss 61.33250694208172  correct 64  time 2487.9 ms
Epoch 600  loss 60.63742567452177  correct 66  time 2631.1 ms
Epoch 610  loss 61.28732627934356  correct 66  time 2546.6 ms
Epoch 620  loss 60.77418775586178  correct 68  time 2481.1 ms
Epoch 630  loss 60.79571214457853  correct 67  time 2476.9 ms
Epoch 640  loss 60.39747255974034  correct 67  time 2597.5 ms
Epoch 650  loss 60.5471222776798  correct 68  time 2543.4 ms
Epoch 660  loss 61.10501458844457  correct 67  time 2485.6 ms
Epoch 670  loss 59.76024457296  correct 69  time 2785.0 ms
Epoch 680  loss 60.22156363006366  correct 67  time 2480.3 ms
Epoch 690  loss 61.468555828677296  correct 66  time 2495.0 ms
Epoch 700  loss 60.24638469716814  correct 66  time 2482.4 ms
Epoch 710  loss 60.08432983310773  correct 66  time 2499.3 ms
Epoch 720  loss 60.415176381421134  correct 67  time 2539.0 ms
Epoch 730  loss 60.11609194776112  correct 64  time 2488.7 ms
Epoch 740  loss 59.717457243185514  correct 65  time 2499.0 ms
Epoch 750  loss 59.946975200271204  correct 65  time 2495.4 ms
Epoch 760  loss 59.91499996371209  correct 68  time 2478.3 ms
Epoch 770  loss 59.78835036061397  correct 66  time 2491.5 ms
Epoch 780  loss 60.227453185716826  correct 67  time 2651.1 ms
Epoch 790  loss 59.1084629008497  correct 64  time 2483.5 ms
Epoch 800  loss 59.1450497251737  correct 69  time 2489.5 ms
Epoch 810  loss 59.23991953904767  correct 67  time 2564.3 ms
Epoch 820  loss 59.41616193051836  correct 67  time 2485.5 ms
Epoch 830  loss 58.90841590829686  correct 67  time 2479.8 ms
Epoch 840  loss 59.23040153535554  correct 66  time 2487.4 ms
Epoch 850  loss 60.28785922339176  correct 65  time 2483.1 ms
Epoch 860  loss 58.10045480354818  correct 66  time 2490.8 ms
Epoch 870  loss 59.41415777723297  correct 65  time 2486.3 ms
Epoch 880  loss 59.24790868535667  correct 66  time 2673.8 ms
Epoch 890  loss 59.13157829265781  correct 67  time 2482.3 ms
Epoch 900  loss 58.88162648485408  correct 66  time 2507.8 ms
Epoch 910  loss 58.58732721456559  correct 67  time 2488.4 ms
Epoch 920  loss 58.21797933854282  correct 68  time 2488.1 ms
Epoch 930  loss 58.966631679037924  correct 67  time 2475.6 ms
Epoch 940  loss 58.15574439393988  correct 66  time 2482.2 ms
Epoch 950  loss 58.84296263266262  correct 66  time 2485.3 ms
Epoch 960  loss 58.433812185598946  correct 67  time 2545.9 ms
Epoch 970  loss 58.179088865056386  correct 67  time 2476.4 ms
Epoch 980  loss 58.430252465590016  correct 66  time 2478.6 ms
Epoch 990  loss 59.23578094728748  correct 66  time 2477.4 ms
Epoch 1000  loss 57.987172931255365  correct 66  time 2476.3 ms
Epoch 1010  loss 58.565392570261935  correct 67  time 2490.7 ms
Epoch 1020  loss 57.68366200300487  correct 66  time 2476.3 ms
Epoch 1030  loss 58.854415665810976  correct 66  time 2479.8 ms
Epoch 1040  loss 57.40205637554633  correct 67  time 2486.5 ms
Epoch 1050  loss 58.96222122933681  correct 65  time 2479.1 ms
Epoch 1060  loss 57.64578220630304  correct 67  time 2516.6 ms
Epoch 1070  loss 57.87213129074868  correct 67  time 2478.8 ms
Epoch 1080  loss 57.77662184208625  correct 67  time 2499.5 ms
Epoch 1090  loss 56.819702980521164  correct 72  time 2476.5 ms
Epoch 1100  loss 56.759263828186654  correct 70  time 2491.3 ms
Epoch 1110  loss 58.265097875435714  correct 67  time 2522.4 ms
Epoch 1120  loss 56.344602437200656  correct 66  time 2485.8 ms
Epoch 1130  loss 58.781585965250585  correct 66  time 2512.5 ms
Epoch 1140  loss 56.07548211655259  correct 73  time 2518.7 ms
Epoch 1150  loss 56.212529627351465  correct 73  time 2505.1 ms
Epoch 1160  loss 56.48584953103177  correct 72  time 2473.3 ms
Epoch 1170  loss 57.27864796003582  correct 72  time 2486.7 ms
Epoch 1180  loss 55.08040842592341  correct 74  time 2495.0 ms
Epoch 1190  loss 55.77012414802532  correct 72  time 2476.2 ms
Epoch 1200  loss 55.19574197581335  correct 73  time 2477.7 ms
Epoch 1210  loss 54.07360531128987  correct 73  time 2478.1 ms
Epoch 1220  loss 62.31101725052846  correct 59  time 2493.7 ms
Epoch 1230  loss 54.55847863476469  correct 70  time 2487.0 ms
Epoch 1240  loss 56.68575219856522  correct 67  time 2547.4 ms
Epoch 1250  loss 54.072729104868145  correct 71  time 2508.0 ms
Epoch 1260  loss 56.36238299054896  correct 69  time 2554.6 ms
Epoch 1270  loss 55.8427257993396  correct 68  time 2551.5 ms
Epoch 1280  loss 57.81825026668004  correct 66  time 2490.6 ms
Epoch 1290  loss 54.667040307779544  correct 68  time 2529.4 ms
Epoch 1300  loss 57.072841603600516  correct 71  time 2501.9 ms
Epoch 1310  loss 54.04492738127171  correct 70  time 2499.3 ms
Epoch 1320  loss 52.44658319753745  correct 69  time 2514.3 ms
Epoch 1330  loss 63.42182882739139  correct 61  time 2542.0 ms
Epoch 1340  loss 56.06025249260475  correct 63  time 2483.2 ms
Epoch 1350  loss 55.64775288759737  correct 65  time 2604.5 ms
Epoch 1360  loss 56.280239205939544  correct 64  time 2500.2 ms
Epoch 1370  loss 55.92414806327096  correct 66  time 2536.0 ms
Epoch 1380  loss 58.07538070439873  correct 63  time 2483.8 ms
Epoch 1390  loss 58.756640774220095  correct 60  time 2565.4 ms
Epoch 1400  loss 54.06741067884519  correct 67  time 2483.8 ms
Epoch 1410  loss 54.67273459676789  correct 68  time 2497.4 ms
Epoch 1420  loss 54.260548995934435  correct 69  time 2558.4 ms
Epoch 1430  loss 52.047053519608134  correct 68  time 2520.7 ms
Epoch 1440  loss 54.75234943889809  correct 68  time 2503.8 ms
Epoch 1450  loss 52.64484671688338  correct 69  time 2505.8 ms
Epoch 1460  loss 52.629612231205456  correct 69  time 2575.2 ms
Epoch 1470  loss 66.62020796331946  correct 52  time 2523.7 ms
Epoch 1480  loss 53.37298957799497  correct 67  time 2488.2 ms
Epoch 1490  loss 53.71900124088128  correct 67  time 2483.7 ms
Epoch 1500  loss 53.846099260940974  correct 66  time 2564.3 ms