# train_Avoid_Propensity
create trained ML algorithm for Avoid Propensity.
## train data
provide training data with the following attributes:
* domain: one of the following domains {'Refrigeration','Washer','Dryer','Cooking','LawnGarden'}
* symptoms
* coverage = {'IW','SP'}
* age: in months
* month = {1,2, ... ,12}
* model
* brand
* 7day_stick = {0,1}

## input data

SELECT TOP 1000
      [mds_dm_ds] as domain
      ,[SES_STA_DT]
      ,MONTH([SES_STA_DT]) as month
      ,[ATN_CD]
      ,SUBSTRING([ATN_CD],7,9) as coverage
      ,[SVC_REQ]
      ,[SO_CVG_CD]
      ,[PM_CHK_FL]
      ,[AVOID_FL]
      ,CASE WHEN [AVD_7D_STICK] is null THEN 0 ELSE AVD_7D_STICK END as result
      ,[prd_mdl_no] as model
      ,[mth_lif_no] as age
      ,[svc_atp_no]
      ,[bnd_nm] as brand
  FROM [Source_data].[reports].[VIEW_AD_funnel]
  Where mds_dm_ds ='Dryer'
  AND (SES_STA_DT BETWEEN '2016-01-01' AND '2016-01-05')
  AND ((ATN_CD IN ('A AVD IW','A AVD SP')) OR ((ATN_CD IN ('A SOC IW','A SOC SP')) AND (SO_CVG_CD IN ('SP','IW'))))
  AND ((PM_CHK_FL is NULL) OR (PM_CHK_FL != 'P'))
  AND ((svc_atp_no is NULL) OR (svc_atp_no < 2))
  AND lower(SVC_REQ) not like '%recall%'



