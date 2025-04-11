# Modeling and Control of a Steam Generator in a Nuclear Power Plant: Ensuring Performance and Safety
####################################################################

AUTHORS’ DECLARATION
Refat Mohamed Abdullah Eshaq declares and claims the following:
1. No one can use and develop this code or merge it with any other codes under any circumstances without explicit permission from the author of this project. If you redistribute the adjusted materials, the author have the right to prosecute individuals, journals, institutions, and even governments.
2. These results are just part of our future work.
3. This work has been built based on the result of the thesis ( ## Ramalho, F.P., 2002. Nonlinear H-infinity control of nuclear steam generators. University of Michigan)
   
####################################################################

The function of a steam generator (SG) in a nuclear power plant is to transfer the heat generated in the reactor core (primary side) to the water that circulates in the secondary side, creating the steam that will drive the turbines. To that end, the hot water coming from the reactor core flows inside the U-tubes (see Figure) and transfers the heat to the water on the secondary side flowing through the shell side or riser, creating a two-phase flow.In the steam separator region, the liquid water is separated from the steam, and recirculated back to the riser through the downcomer. The steam generated accumulates inside the steam dome and flows outside through a series of valves to drive the turbines. The water level is measured inside the downcomer and has to be kept within some pre-defined range for proper operation of the steam generator (Ramalho, F.P., 2002. Nonlinear H-infinity control of nuclear steam generators. University of Michigan).

![_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=5868030111270194911 skey=@crypt_15c29c41_6d33599d9dcb47f8d2d1cf4889b52f53 mmweb_appid=wx_webfilehelper](https://github.com/user-attachments/assets/53235855-a946-4a53-86c9-98008105fe5f)

## The Importance and Problems of Water Level Control 

The control of the water level in the downcomer region is of the highest importance for the proper working of a nuclear power plant. Around 25% of all the downtime of the commercial plants is due to problems related to the SG water level. To understand why it is so hard in practice to control the water level, we have to realize that the SG can show inverse responses due to the shrink-and-swell effect, where the indicated change in liquid mass inside the SG, based on the water level measurement, is contrary to the actual change in the water inventory. This effect is basically due to the presence of the two-phase flow inside the riser. and can occur when the feedwater or the steam flow increases or decreases (Ramalho, F.P., 2002. Nonlinear H-infinity control of nuclear steam generators. University of Michigan).

## Mass balances 

![Capture11](https://github.com/REFATESHAQ/Steam-generator-nuclear-power-/assets/48349737/ef83c4c1-cb08-4e7a-89b5-7604323a5ffa)

(Ramalho, F.P., 2002. Nonlinear H-infinity control of nuclear steam generators. University of Michigan)

### The final model is to study the behavior of the steam quality and water level of the steam generator. Now, we want to design control for this problem (shrink-and-swell).

![Capture](https://github.com/REFATESHAQ/Steam-generator-nuclear-power-/assets/48349737/02716de4-33a0-4222-8d06-b4811d3ef674)
