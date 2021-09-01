using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class Handpresence : MonoBehaviour
{
    public bool showController = false;
    public InputDeviceCharacteristics controllerCharacteristics;
    private InputDevice targetDevice;
    public List<GameObject> controllerPrefabs;
    private GameObject spawnedController;
    public GameObject handModelPrefab;
    private GameObject spawnedhandModel;
    private Animator handAnimator;


    // Start is called before the first frame update
    void Start()
    {
        TryInitialize();

    }
    void TryInitialize()
    {
        List<InputDevice> devices = new List<InputDevice>();

        InputDevices.GetDevicesWithCharacteristics(controllerCharacteristics, devices);


        if (devices.Count > 0)
        {
            targetDevice = devices[0];
            GameObject prefab = controllerPrefabs.Find(controller => controller.name == targetDevice.name);
            if (prefab)
            {
                spawnedController = Instantiate(prefab, transform);
            }
            else
            {
                spawnedController = Instantiate(controllerPrefabs[0], transform);
            }

            spawnedhandModel = Instantiate(handModelPrefab, transform);
            handAnimator = spawnedhandModel.GetComponent<Animator>();
        }
    }


    void UpdateHandAnimation()
    {
        if (targetDevice.TryGetFeatureValue(CommonUsages.trigger, out float triggerValue))
        {
            handAnimator.SetFloat("Trigger", triggerValue);
        }
        else
        {
            handAnimator.SetFloat("Trigger", 0);
        }

        if (targetDevice.TryGetFeatureValue(CommonUsages.grip, out float gripValue))
        {
            handAnimator.SetFloat("Grip", gripValue);
        }
        else
        {
            handAnimator.SetFloat("Grip", 0);
        }
    }
    // Update is called once per frame
    void Update()
    {
        if (!targetDevice.isValid)
        {
            TryInitialize();
        }
        else
        {
            if (showController)
            {
                spawnedController.SetActive(true);
                spawnedhandModel.SetActive(false);
            }
            else
            {
                spawnedController.SetActive(false);
                spawnedhandModel.SetActive(true);
                UpdateHandAnimation();
            }
        }

    }
}

