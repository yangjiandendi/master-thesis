using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class locomotionController : MonoBehaviour
{
    public XRController rightTeleportRay;
    public XRController leftTeleportRay;
    public InputHelpers.Button teleportActivationButton;
    public float activationthreshold = 0.1f;


    // Update is called once per frame
    void Update()
    {
        if (rightTeleportRay)
        {
            rightTeleportRay.gameObject.SetActive(CheckIfActivated(rightTeleportRay));
        }
        if (leftTeleportRay)
        {
            leftTeleportRay.gameObject.SetActive(CheckIfActivated(leftTeleportRay));
        }
    }

    public bool CheckIfActivated(XRController controller)
    {
        InputHelpers.IsPressed(controller.inputDevice, teleportActivationButton, out bool isActivated, activationthreshold);
        return isActivated;
    }
}
