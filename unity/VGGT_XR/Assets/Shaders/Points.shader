Shader "VGGTXR/Points"
{
    Properties { _PointPixels ("Point pixels", Float) = 3 }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            ZWrite On Cull Off
            CGPROGRAM
            #pragma target 4.0
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            #include "UnityCG.cginc"
            float _PointPixels;
            struct Input { float4 vertex:POSITION; float4 color:COLOR; };
            struct Output { float4 position:SV_POSITION; float4 color:COLOR; };
            Output vert(Input v) { Output o; o.position=UnityObjectToClipPos(v.vertex); o.color=v.color; return o; }
            [maxvertexcount(4)]
            void geom(point Output p[1], inout TriangleStream<Output> stream)
            {
                float2 corners[4]={float2(-1,-1),float2(-1,1),float2(1,-1),float2(1,1)};
                for(int i=0;i<4;i++){Output o=p[0];o.position.xy+=corners[i]*_PointPixels/_ScreenParams.xy*o.position.w;stream.Append(o);}
            }
            float4 frag(Output i):SV_Target { return i.color; }
            ENDCG
        }
    }
}
