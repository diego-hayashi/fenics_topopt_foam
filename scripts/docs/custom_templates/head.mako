## ⭐️ Custom styling for the home link 
## https://github.com/pdoc3/pdoc/issues/71
<%! 
     from pdoc.html_helpers import minify_css 
 %> 

<%def name="homelink()" filter="minify_css">

	/* Home link */
	.homelink {

		/* Display */
		display: block;

		/* Font configurations */
		font-size: 1em;
		font-weight: bold;
		color: #555;

		/* padding-bottom: .5em; */
		/* border-bottom: 1px solid silver; */

		/* Margins */
		margin-left: auto;
		margin-right: auto;

	}
	.homelink:hover {
		color: inherit;
		background-color: inherit;
	}
	.homelink img {

		/* Display */
		display: block;

		/* Image size */
		max-height: 5em;

		/* Margins */
		margin-left: auto;
		margin-right: auto;
		/* margin-bottom: .3em; */
	}

	.name_text2 {
			
		/* Font configurations */
		font-size: 1.5em;
		text-align: center;
		font-weight: bold;
		color: gray;

		/* Margins */
		margin-top: .3em;

	}

	.name_text {
			
		/* Font configurations */
		font-size: 1.5em;
		text-align: center;
		font-weight: bold;
		color: gray;

	}

	.name_text_small {
			
		/* Font configurations */
		font-size: 1.3em;
		text-align: center;
		font-weight: normal;
		color: #2e3436;

		/* Background color */
		/* background-color: #bad5ff; */
		/* background-color: #eeeeee; */
		/* background-color: #c5daf4; */
		/* https://stackoverflow.com/questions/23201134/transparent-argb-hex-value */
		background-color: #c5daf4cc;
		border-radius: 10px;
		/* opacity: 0.80; */

		/* Margins */
		padding-top: .3em;
		padding-bottom: .3em;
	}

	/* hr_gray */
	.hr_gray {

		/* Size */
		height: 2px;

		/* Color */
		color: #c0bfbc; /* gray; */

	}

	/* hr_gray2 */
	.hr_gray2 {

		/* Size */
		height: 2px;

		/* Color */
		color: #c0bfbc; /* gray; */

		/* Margins */
		margin-bottom: 0em;
	}

	.custom_button {

		/* Cursor */
		cursor: pointer;

		/* Display */
		display: inline-block;

		/* Font configurations */
		/* font-size: 0.8em; */
		color: black;

		/* Border and padding */
		border: none;
		padding: 0px 0px;

		/* Background */
		background-color: inherit;

		/* Brightness */
		/* filter: brightness(100%); */

		/* Opacity */
		opacity: 1.0;
		/* opacity: 0.8; */

		/* Gray text */
		/* https://stackoverflow.com/questions/55220093/grey-out-emoji-characters-html-css/55220163 */
		/* filter: grayscale(100%); */

		/* text-shadow: 2px 2px 5px black; */

		/* Border */
		border-radius: 8px;

		/* Margins */
		padding-left: .3em;
		padding-right: .3em;
		padding-bottom: .3em;
		padding-top: .3em;

	}

	.custom_button:hover {

		/* Font configurations */
		color: black;

		/* Opacity */
		opacity: 1.0;
		/* opacity: 0.98; */

		/* Brightness */
		/* filter: brightness(105%); */

		/* Background */
		/* background-color: inherit; */
		/* background-color: white; */
		background-color: #f6faff;

		/* text-shadow: 2px 2px 1px black; */

	}

	.custom_button:active {

		/* Background */
		/* background-color: inherit; */
		/* background-color: #d3d7cf; */
		background-color: #a1cbff;

	}

	.custom_button2 {

		/* Cursor */
		cursor: pointer;

		/* Display */
		display: inline-block;

		/* Font configurations */
		/* font-size: 0.8em; */
		color: black;

		/* Border and padding */
		border: none;
		padding: 0px 0px;

		/* Background */
		/* background-color: inherit; */
		background-color:rgba(0, 0, 0, 0.0);

		/* Brightness */
		/* filter: brightness(100%); */

		/* Opacity */
		opacity: 0.53;

		/* Gray text */
		/* https://stackoverflow.com/questions/55220093/grey-out-emoji-characters-html-css/55220163 */
		/* filter: grayscale(100%); */

		/* text-shadow: 2px 2px 5px black; */

		/* Border */
		border-radius: 8px;

		/* Margins */
		padding-left: .3em;
		padding-right: .3em;
		padding-bottom: .3em;
		padding-top: .3em;

	}

	.custom_button2:hover {

		/* Font configurations */
		color: black;

		/* Opacity */
		opacity: 1.0;
		/* opacity: 0.98; */

		/* Brightness */
		/* filter: brightness(105%); */

		/* Background */
		/* background-color: inherit; */
		/* background-color: white; */
		/* background-color: #c5daf4; */ /* #f6faff; */

		/* text-shadow: 2px 2px 1px black; */

	}

	.custom_button2:active {

		/* Opacity */
		opacity: 1.0;
		/* opacity: 0.98; */

		/* Background */
		/* background-color: inherit; */
		/* background-color: #d3d7cf; */
		/* background-color: #99c1f1; */

	}

	/* ---------------- Basic tooltip with an arrow --------------------- */
	/* * Pop-up-like tooltip */

	.arrowtooltip {
		position: relative;
		display: inline-block;
	}

	.arrowtooltip .tooltiptext {
		visibility: hidden;
		width: auto;
		background-color: #2e3436;
		color: #fff;
		text-align: center;
		border-radius: 6px;
		padding: 10px 10px;

		/* Fixed font-size for the tooltip */
		font-size: 20px;

		/* Position the tooltip */
		position: absolute;
		z-index: 1;
		top: 150%; /* Tooltip below */
		left: 50%;

		/* Center tooltip horizontally */
		/* https://stackoverflow.com/questions/21709674/how-to-center-a-tooltip-horizontally-on-top-of-a-link */
		transform: translateX(-50%);

		/* Don't ommit whitespaces */
		white-space:nowrap;

		/* Fade in tooltip - takes 0.25s to go from 0% to 100% opac: */
		opacity: 0;
		transition: opacity 0.25s;
	}

	.arrowtooltip .tooltiptext::after {
		content: "";
		position: absolute;
		bottom: 100%; /* Triangle below */
		left: 50%;
		margin-left: -5px;

		/* Triangle */
		border-width: 5px;
		border-style: solid;
		border-color: transparent transparent #2e3436 transparent; /* Triangle below */
	}

	.arrowtooltip:hover .tooltiptext {
	/*	opacity: 1;*/
		opacity: 0.95;
		visibility: visible;
	}

	.arrowtooltip2 {
		position: relative;
		display: inline-block;
	}

	.arrowtooltip2 .tooltiptext {
		visibility: hidden;
		width: auto;
		background-color: #2e3436;
		color: #fff;
		text-align: center;
		border-radius: 6px;
		padding: 10px 10px;
		font-weight: normal;

		/* Fixed font-size for the tooltip */
		font-size: 20px;

		/* Position the tooltip */
		position: absolute;
		z-index: 1;
		top: 130%; /* Tooltip below */
		/* top: 120%; */ /* Tooltip below */
		left: 50%;

		/* Center tooltip horizontally */
		/* https://stackoverflow.com/questions/21709674/how-to-center-a-tooltip-horizontally-on-top-of-a-link */
		transform: translateX(-50%);

		/* Don't ommit whitespaces */
		white-space:nowrap;

		/* Fade in tooltip - takes 0.25s to go from 0% to 100% opac: */
		opacity: 0;
		transition: opacity 0.25s;
	}

	.arrowtooltip2 .tooltiptext::after {
		content: "";
		position: absolute;
		bottom: 100%; /* Triangle below */
		left: 50%;
		margin-left: -5px;

		/* Triangle */
		border-width: 5px;
		border-style: solid;
		border-color: transparent transparent #2e3436 transparent; /* Triangle below */
	}

	.arrowtooltip2:hover .tooltiptext {
	/*	opacity: 1;*/
		opacity: 0.95;
		visibility: visible;
	}

</%def>

<style>
	## ⭐️ Preserve newlines from source code docstrings
	## https://github.com/pdoc3/pdoc/pull/38
	## https://github.com/mitmproxy/pdoc/issues/179#issuecomment-466465655
	dd p {
		white-space: pre-wrap;
	}

	## ⭐️ Custom styling for the home link
	## https://github.com/pdoc3/pdoc/issues/71
	${homelink()}

</style>

## ⭐️ Add a favicon
## https://github.com/pdoc3/pdoc/blob/master/doc/pdoc_template/head.mako
<link rel="icon" href="fenics_topopt_foam_favicon.png">

