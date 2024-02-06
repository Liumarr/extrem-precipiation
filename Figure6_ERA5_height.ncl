;;;;;;;;;;;;;;;;;;;;;;
; FNL precip and precip type
; 2022-2-23
;;;;;;;;;;;;;;;;;;;;;

	;;;;;;;;;;;  read  ERA5 data
	;cp      convective precipation
	;crr     convective rain rate
	;msl     mean sealevel pressure
	;sst     sea surface temperature
	;sp      surface pressure 
	;tciw    total cloud ice
	;tclw    total cloud rain
	;tp      total precipation
	;;;;;;;;;;;;;;;;;;
	
    dir      = "../data/"
    f        = addfile("../data/era5/adaptor.mars.internal-1703558852.43509-25958-18-ebfcb787-346a-4c86-a95e-45d4bbf18c44.nc","r")


	lat        = f->latitude
	lon        = f->longitude	
	;print(level)
	time       = cd_calendar(f->time,-3)
	print(time)

	; u950          = short2flt(f->u(:,{925},:,:))
	; v950          = short2flt(f->v(:,{925},:,:))	
	u850          = short2flt(f->u(:,{925},:,:))
	v850          = short2flt(f->v(:,{925},:,:))
	u500          = short2flt(f->u(:,{500},:,:))
	v500          = short2flt(f->v(:,{500},:,:))
	; u200          = short2flt(f->u(:,{200},:,:))
	; v200          = short2flt(f->v(:,{200},:,:))
	z200          = short2flt(f->z(:,{200},:,:))
    z500          = 0.1*short2flt(f->z(:,{500},:,:))/9.8
    ; z925          = short2flt(f->z(:,{925},:,:))
	; t200          = short2flt(f->t(:,{200},:,:))
    ; t500          = short2flt(f->t(:,{500},:,:))
    ; t850          = short2flt(f->t(:,{850},:,:))
    ; t925          = short2flt(f->t(:,{925},:,:))
    ; q925          = short2flt(f->q(:,{925},:,:))
    crwc          = 1000*dim_sum_n(short2flt(f->crwc(:,:,:,:)),1)
    ; w950          = short2flt(f->w(:,{925},:,:))
    moment        = 4

   copy_VarCoords(z200,crwc)
	copy_VarCoords(z200,z500)
    ; z200          = 0.1*z200/9.8
    ; copy_VarCoords(z500,z200)     
	plotType = "pdf"       ; send graphics to PNG file
    plotName = "Figure1-a-xingtai"
	; plot     = new(2,graphic)
	wks 	 = gsn_open_wks(plotType, plotName) 


	res                      = True
	res@gsnFrame             = False               ; don't advance frame
	res@gsnDraw              = False               ; don't draw plot
    res@gsnMaximize          = False
    res@gsnStringFont        = 25
    
	res@vpWidthF             = 0.4
	res@vpHeightF            = 0.3
	res@gsnAddCyclic         = False    ; Data is not cyclic	
	res@gsnLeftString        = ""
	res@gsnRightString       = ""	
	res@gsnLeftStringFontHeightF =0.014
	res@tmXBLabelFont           = 25
    res@tmYLLabelFont           = 25
	res@tmXBLabelFontHeightF    = 0.014
	res@tmYLLabelFontHeightF    = 0.014
	res@tmXBMajorOutwardLengthF = -0.001
	res@tmXBMinorOutwardLengthF = -0.001
	res@tmYLMajorOutwardLengthF = -0.001
	res@tmYLMinorOutwardLengthF = -0.001
	res@tmYRMajorOutwardLengthF = -0.001
	res@tmYRMinorOutwardLengthF = -0.001

    rest  = res
    vcres = res
	vcres@vcGlyphStyle               = "WindBarb";"FillArrow"
    vcres@vcWindBarbScaleFactorF     = 2
	vcres@vcMinDistanceF             = 0.02
    vcres@vcRefMagnitudeF            = 4.   
    vcres@vcWindBarbCalmCircleSizeF  = 0.1
    ;vcres@vcGlyphOpacityF            = 0.7   
    ;vcres@vcWind          
    vcres@vcRefLengthF               = 0.03 
    vcres@vcFillArrowEdgeColor       = "Transparent"  
    vcres@vcFillArrowMinFracWidthF   = 0.02
    ;vcres@vcWindBarbLineThicknessF   = 2.0
    vcres@vcRefAnnoOn                = False        
    ;vcres@vcRefAnnoString1           = "~F25~10 m/s"
    vcres@vcRefAnnoFontHeightF       = 0.01
    vcres@vcRefAnnoSide              = "Top"
    vcres@vcRefAnnoString2On         = False
    vcres@vcRefAnnoPerimOn           = False
    vcres@vcRefAnnoOrthogonalPosF    = -0.16
    ;vc@vcRefAnnoParallelPosF      = 0.96
    vcres@vcRefAnnoBackgroundColor   = "Transparent"

    res@cnInfoLabelOn	            = False
	res@mpDataBaseVersion           = "MediumRes"          ; Higher res coastline
	res@mpGridLineDashPattern       = 15
	res@mpGridLineColor             = "black"
	res@mpLimitMode                 = "LatLon"
	; res@mpMinLatF           = 32
	; res@mpMaxLatF           = 42
	; res@mpMinLonF           = 110
	; res@mpMaxLonF           = 120
    res@mpMinLatF           = 20
	res@mpMaxLatF           = 50
	res@mpMinLonF           = 90
	res@mpMaxLonF           = 130
    res@mpFillOn            = False
	res@pmTickMarkDisplayMode = "Always"
    res@cnLinesOn           = True
    res@cnFillOn            = False
    res@cnLineLabelPlacementMode = "Constant"
	res@cnLineThicknessF    = 2.5
    res@cnLineLabelsOn      = True
    res@cnLineLabelDensityF = 0.001
    res@gsnLeftStringFontHeightF = 0.014

    res@cnLevelSelectionMode = "ExplicitLevels"
    res@cnLevels            = ispan(572,592,2)
    res@cnLineLabelFont     = 25
    res@cnLineLabelInterval = 2
    res@cnLineDashSegLenF   = 0.2
    res@cnLineLabelsOn      = True
    res@cnLineLabelStrings  = tostring(ispan(572,592,2))

	plot                    = gsn_csm_contour_map(wks,z500(moment,:,:),res)
    res@mpMinLatF           = 30
	res@mpMaxLatF           = 40
	res@mpMinLonF           = 110
	res@mpMaxLonF           = 120
    res@cnLinesOn           = False
    res@cnFillOn            = True
    res@lbLabelBarOn        = True
    res@cnLevels            := 0.1*ispan(0,40,2)
    res@cnFillPalette       = "nice_gfdl"
    res@lbBoxLinesOn        = True
    res@lbBoxLineThicknessF = 0.1
    res@lbLabelFont         = 25
    res@lbLabelStride       = 4
    res@pmLabelBarHeightF   = 0.05
    res@lbTitlePosition     = "bottom"
    res@lbTitleFontHeightF  = 0.01
    res@lbTitleString       = "~F25~RWC (g / kg)"
    res@lbBoxEndCapStyle  = "TriangleBothEnds"
    res@lbBoxLineThicknessF   = 1.1

    plot2                 = gsn_csm_contour_map(wks,crwc(moment,:,:),res)
    ; res@lbTitleString       = "~F25~Vertical Velocity (Pa / s)"
    ; res@cnFillPalette       := "MPL_bwr"
    ; res@cnLevels            := -0.05*ispan(-24,24,3)
    ; plot(1)                 = gsn_csm_contour_map(wks,w950(moment,:,:),res)
    
	; vc0              = gsn_csm_vector(wks,u950(moment,:,:),v950(moment,:,:),vcres)
	vc1              = gsn_csm_vector(wks,u850(moment,:,:),v850(moment,:,:),vcres)
	;vc                = gsn_csm_vector(wks,u500(moment,:,:),v500(moment,:,:),vcres)
	;vc3              = gsn_csm_vector(wks,u200(moment,:,:),v200(moment,:,:),vcres)

    ; rest@cnLineColor        = "red"
    ; rest@cnLevelSelectionMode = "ExplicitLevels"
    ; rest@cnLevels            = ispan(240, 300, 1)
    ; rest@cnLinesOn           = False
    ; rest@cnFillOn            = True
    ; rest@cnLineLabelPlacementMode = "Constant"
    ; rest@cnInfoLabelOn	         = False
	; rest@cnLineThicknessF    = 2.
    ypts = (/38.5, 38.5, 36.5, 36.5, 38.5/)
    xpts = (/116, 114, 114, 116, 116/)
   ;************************************************
   ; add the box
   ;************************************************
    resp                  = True                      ; polyline mods desired
    resp@gsLineColor      = "grey35"                     ; color of lines
    resp@gsLineThicknessF = 3.0                       ; thickness of lines
   
   ; create array of dummy graphic variables. This is required, b/c each line
   ; must be associated with a unique dummy variable.
   
     dum = new(4,graphic)  
   
   ; draw each line separately. Each line must contain two points.
     do i = 0, 3
       dum(i)=gsn_add_polyline(wks,plot,xpts(i:i+1),ypts(i:i+1),resp)      
     end do 
   
    ; cnt              = gsn_csm_contour(wks,t200(moment,:,:),rest)
    ; cnt1             = gsn_csm_contour(wks,t500(moment,:,:),rest)
    ; cnt2             = gsn_csm_contour(wks,t850(moment,:,:),rest)
    ; cnt3             = gsn_csm_contour(wks,t925(moment,:,:),rest)
	overlay(plot,vc1)

    ; draw(plot)
    ; frame(wks)
    ; overlay(plot(2),vc1)
    ; overlay(plot(3),vc0)
	; overlay(plot(0),cnt)
    ;overlay(plot(1),cnt1)
    ; overlay(plot(2),cnt2)
    ; overlay(plot(3),cnt3)



	; point = True
	; point@gsMarkerThicknessF = 2
	; point@gsMarkerIndex    = 7
	; point@gsMarkerSizeF    = 0.01
    ; point@gsMarkerThicknessF = 3
	; point@gsMarkerColor    = "royalblue"
	; dot   = gsn_add_polymarker(wks, plot(0), 114.7, 37.13, point)
    ; dot1   = gsn_add_polymarker(wks, plot(1), 114.7, 37.13, point)
    ; dot2   = gsn_add_polymarker(wks, plot(2), 114.7, 37.13, point)
    ; dot3   = gsn_add_polymarker(wks, plot(3), 114.7, 37.13, point)
	; resP                  = True
    ; resP@gsnPanelTop         = 0.99
	; resP@gsnPanelBottom      = 0.1
	; resP@gsnPanelLeft        = 0.05
    ;resP@gsnMaximize      = True
	
    ; lnres = True
    ; lnres@gsLineColor = "black"
    ; lnres@gsLineThicknessF = 1
    ; poly = gsn_add_shapefile_polylines(wks,plot(0),"data/chinaMap/China_GuoJieShengJie_Polyline.shp",lnres)
    ; poly1 = gsn_add_shapefile_polylines(wks,plot(1),"data/chinaMap/China_GuoJieShengJie_Polyline.shp",lnres)
    ; poly2 = gsn_add_shapefile_polylines(wks,plot(2),"data/chinaMap/China_GuoJieShengJie_Polyline.shp",lnres)
    ; poly3 = gsn_add_shapefile_polylines(wks,plot(3),"data/chinaMap/China_GuoJieShengJie_Polyline.shp",lnres)
    resp                      = True
	resp@gsnFrame             = False               ; don't advance frame
	gsn_panel(wks,(/plot,plot2/),(/1,2/),resp)

	
	
	
	