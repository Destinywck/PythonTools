<code name="nedatasync_stask.cpp">
	<function name="__GetNeedSyncNANE__YAXHAAV__vector_KV__allocator_K_std___std___Z.dot">
		<buggy matched="true" nums="2">
			<match num="1">
				<pair buggy="" new="" />
				<pair buggy="buggy.cpp:29:40" new="nedatasync_stask.cpp:916:14" />
				<pair buggy="buggy.cpp:29:40" new="nedatasync_stask.cpp:916:14" />
				<pair buggy="buggy.cpp:29:54" new="nedatasync_stask.cpp:916:25" />
			</match>
			<match num="2">
				<pair buggy="" new="" />
				<pair buggy="buggy.cpp:29:40" new="nedatasync_stask.cpp:918:115" />
				<pair buggy="buggy.cpp:29:40" new="nedatasync_stask.cpp:918:115" />
				<pair buggy="buggy.cpp:29:54" new="nedatasync_stask.cpp:918:126" />
			</match>
		</buggy>
		<fine matched="true" nums="1">
			<match num="1">
				<pair fine="" new="" />
				<pair fine="fine.cpp:28:12" new="nedatasync_stask.cpp:914:10" />
				<pair fine="fine.cpp:28:7" new="nedatasync_stask.cpp:914:8" />
				<pair fine="fine.cpp:29:10" new="nedatasync_stask.cpp:916:14" />
				<pair fine="fine.cpp:29:10" new="nedatasync_stask.cpp:916:14" />
				<pair fine="fine.cpp:29:24" new="nedatasync_stask.cpp:916:25" />
			</match>
		</fine>
		<result buggy="false" fine="true" />
	</function>
	
Target product,CVE-ID,"Vulnerable product
published in NVD",Vulnerability publish time,Vulnerability location in target product,"Library/API
function call",1st patched version of target product,Date of 1st patched version,paper
Thunderbird 24.8.0,CVE-2015-0834,,2015/2/25,.../PeerConnectionImpl.cpp,,Thunderbird 38.0.1,2015/6/11,2016-VulPecker
,CVE-2014-8643,,2015/1/14,.../nsEmbedFunctions.cpp,,Thunderbird 38.3.0,2015/9/29,
,CVE-2014-1498,,2014/3/19,.../src/nsCrypto.cpp,,Thunderbird 31.0,2014/7/22,
,CVE-2013-6167,,2014/2/15,.../nsCookieService.cpp,,Thunderbird 38.4.0,2015/11/24,
Libav 10.1,CVE-2014-9604,,2015/1/16,.../libavcodec/utvideodec.c,,Libav 10.6,2015/3/11,
,CVE-2014-8547,,2014/11/5,.../libavcodec/gifdec.c,,Libav 10.6,2015/3/11,
,CVE-2014-8541,,2014/11/5,.../libavcodec/mjpegdec.c,,Libav 10.6,2015/3/11,
,CVE-2014-2098,,2014/3/1,.../libavcodec/wmalosslessdec.c,,Libav 10.4,2014/8/18,
,CVE-2013-7011,,2013/12/9,.../libavcodec/ffv1dec.c,,Libav 10.4,2014/8/18,
,CVE-2013-7008,,2013/12/9,.../libavcodec/h264.c,,Libav 11.1,2014/12/2,
,CVE-2013-3674,,2013/6/9,.../libavcodec/cdgraphics.c,,Libav 10.4,2014/8/18,
,CVE-2013-0851,,2013/12/7,.../libavcodec/eamad.c,,Libav 10.3,2014/8/4,
Xen 4.4.0,CVE-2014-2894,,2014/4/23,.../ide/core.c,,Xen 4.4.3, 2015/8/25,
,CVE-2014-5263,,2014/8/26,.../usb/hcd-xhci.c,,Xen 4.4.3,2015/8/25,
,CVE-2013-6399,,2014/11/4,.../virtio/virtio.c,,Xen 4.4.3,2015/8/25,
,CVE-2013-4534,,2014/11/4,.../intc/openpic.c,,Xen 4.5.0,2015/1/14,
,CVE-2013-4533,,2014/11/4,.../arm/pxa2xx.c,,Xen 4.5.0,2015/1/14,
,CVE-2013-4530,,2014/11/4,.../ssi/pl022.c,,Xen 4.5.0,2015/1/14,
,CVE-2013-4527,,2014/11/4,.../timer/hpet.c,,Xen 4.5.0,2015/1/14,
,CVE-2013-4151,,2014/11/4,.../virtio/virtio.c,,Xen 4.4.3,2015/8/25,
,CVE-2013-4150,,2014/11/4,.../net/virtio-net.c,,Xen 4.4.3,2015/8/25,
,CVE-2013-4149,,2014/11/4,.../net/virtio-net.c,,Xen 4.4.3,2015/8/25,
Xen 4.6.0,CVE-2016-9104,Qemu,2016/12/9,.../qemu-xen/hw/9pfs/virtio-9p.c,memcpy,Xen 4.9.0,,2018-VulDeePecker
Seamonkey 2.31,CVE-2015-4517,Firefox,2015/9/24,.../system/gonk/NetworkUtils.cpp,snprintf,Seamonkey 2.38,,
,CVE-2015-4513,Firefox,2015/11/5,.../protocol/http/Http2Stream.cpp,memset,Seamonkey 2.39,,
Libav 10.2,CVE-2014-2263,Ffmpeg,2014/2/28,libavformat/mpegtsenc.c,"strchr, strlen",Libav 10.4,,
,CVE-2013-4513,Linux 3.11.4,,,copy_from_user,,,2015-Automatic
,CVE-2014-0160,OpenSSL 1.1.0f,,,memcpy,,,
,CVE-2013-6482,Pidgin 2.10.7,,,atoi,,,
,CVE-2012-3377,VLC 2.0.1,,,memcpy,,,
,CVE-2013-4473,Poppler (Xpdf) 0.24.1,,,sprintf,,,
,CVE-2010-3429,Ffmpeg,,,,,,2012-Generalized
,CVE-2011-4601,Pidgin,,,,,,
,CVE-2006-3459,LibTIFF,,,,,,
,CVE-2011-2529,Asterisk,,,,,,
