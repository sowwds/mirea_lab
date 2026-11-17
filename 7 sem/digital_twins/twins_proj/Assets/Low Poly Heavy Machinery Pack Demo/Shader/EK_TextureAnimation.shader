Shader "Custom/TextureAnimation"
{
    Properties
    {
        _Color("Color", Color) = (1, 1, 1, 1)
        _MainTex("Main Texture", 2D) = "white" {}
        _Tiling("Tiling", Vector) = (1, 1, 0, 0)
        _Speed("Speed", Float) = 1
        _Rotation("Rotation", Float) = 0
    }

        SubShader
        {
            Tags { "RenderType" = "Opaque" }
            LOD 100

            Pass
            {
                CGPROGRAM
                #pragma vertex vert
                #pragma fragment frag
                #pragma target 3.0
                #pragma multi_compile_fog

                #include "UnityCG.cginc"

                struct appdata_t
                {
                    float4 vertex : POSITION;
                    float2 uv : TEXCOORD0;
                };

                struct v2f
                {
                    float2 uv : TEXCOORD0;
                    float4 vertex : SV_POSITION;
                };

                sampler2D _MainTex;
                float4 _MainTex_ST;
                float4 _Tiling;
                float _Speed;
                float _Rotation;
                float4 _Color;

                v2f vert(appdata_t v)
                {
                    v2f o;
                    o.vertex = UnityObjectToClipPos(v.vertex);
                    o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                    return o;
                }

                fixed4 frag(v2f i) : SV_Target
                {
                    fixed2 offset = fixed2(_Speed * _Time.y, 0); // Hareket için offset hesaplama
                    float2 rotatedUV = i.uv;
                    // Texture'ı döndürmek için matris dönüşü kullanma
                    rotatedUV -= 0.5; // Orjini merkeze al
                    float s = sin(_Rotation);
                    float c = cos(_Rotation);
                    float2x2 rotationMatrix = float2x2(c, -s, s, c);
                    rotatedUV = mul(rotationMatrix, rotatedUV);
                    rotatedUV += 0.5; // Geri al

                    fixed4 col = tex2D(_MainTex, rotatedUV + offset) * _Color; // Offset uygulama
                    return col;
                }
                ENDCG
            }
        }
}
