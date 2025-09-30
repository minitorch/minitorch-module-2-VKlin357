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

