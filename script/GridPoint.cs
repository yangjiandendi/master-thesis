using UnityEngine;

public class GridPoint : MonoBehaviour
{
    #region --- events ---
    public delegate void PointValueChange(ref GridPoint gp);
    public static event PointValueChange OnPointValueChange;
    #endregion

    private Vector3 _position = Vector3.zero;
    private float _value = 0.1f;
    private float _size = 0.1f;
    private bool _on = true;
    private float _color = 0.5f;
    private bool _glow = false;

    public Vector3 Position
    {
        get
        {            
            return _position;
        }
        set
        {
            _position = new Vector3(value.x, value.y, value.z);
            if (this != null) 
                this.transform.localPosition = _position;
        }
    }
    public float Value
    {
        get
        {
            return _value;
        }
        set
        {
            _value = value;
        }
    }
    public float Size
    {
        get
        {
            return _size;
        }
        set
        {
            _size = value;
            if (this != null)
                this.transform.localScale = new Vector3(_size, _size, _size);
        }
    }
    public bool On
    {
        get
        {
            return _on;
        }
        set
        {
            _on = value;
            if (this != null)
            {
                Renderer r = this.GetComponent<Renderer>();
                if (r != null)
                    r.enabled = _on;
            }
        }
    }
    public float Color
    {
        get
        {
            return _color;
        }
        set
        {
            _color = value;
            if (this != null)
            {
                Renderer r = this.GetComponent<Renderer>();
                if (r != null)
                    r.material.color = new Color(_color, _color, _color);
            }
        }
    }    
    public bool Glow
    {
        get
        {
            return _glow;
        }
        set
        {
            _glow = value;
            if (this != null)
            {
                Renderer r = this.GetComponent<Renderer>();
                if (r != null)
                {
                    if (_glow == true)
                        r.material.color = new Color(_color, 0, 0);     //glow reddish
                    else
                        r.material.color = new Color(_color, _color, _color);
                }
            }
        }
    }

    private void OnTriggerStay(Collider other)
    {
        this.Value = Mathf.Clamp(this.Value - 0.1f * Time.deltaTime, 0f, 1f);
        this.Color = this.Value;

        if (OnPointValueChange != null)
        {
            GridPoint gp = this.GetComponent<GridPoint>();
            OnPointValueChange(ref gp);
        }
    }
    public override string ToString()
    {
        return string.Format("{0} {1}", Position, Value);
    }
}
