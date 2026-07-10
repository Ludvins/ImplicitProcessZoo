# Models

All supported models expose `predict_f_samples` and `predict_y_samples` using
the `[S,N,D]` contract. Model constructors remain method-specific.

## MAP

::: implicit_process_zoo.map_baseline.DeterministicMAP
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - fit

## MFVI

::: implicit_process_zoo.mfvi.mfvi.MFVI
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - fit

## FBNN

::: implicit_process_zoo.fbnn.fbnn.FBNN
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - fit

## TFSVI

::: implicit_process_zoo.tfsvi.tfsvi.TFSVI
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - fit

## VIP

::: implicit_process_zoo.vip.vip.VIP
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - fit

## FTIP

::: implicit_process_zoo.ftip.ftip.FTIP
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - warm_start_from_vip
        - fit

### Unified FTIP

::: implicit_process_zoo.ftip.ftip.UnifiedFTIP
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - fit

## GMVIP

::: implicit_process_zoo.gmvip.gmvip.GeneralizedMatheronVIP
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - predict_summary

## SIP

::: implicit_process_zoo.sip.sip.SIP
    options:
      members:
        - predict_f_samples
        - predict_y_samples
        - fit
