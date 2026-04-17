import { useEffect, useRef } from 'react';
import '../HalfCourt.css';
import { SHOTCHART_SETTINGS, NBA_SETTINGS } from '../lib/Constants';
import { drawCourt } from '../lib/Utilities';

type HalfCourtProps = {
  handlerXY?: [number, number] | null;
};

const HalfCourt = ({ handlerXY }: HalfCourtProps) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const handlerMarkerRef = useRef<SVGCircleElement | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    svgRef.current.innerHTML = '';
    const chartSettings = SHOTCHART_SETTINGS(NBA_SETTINGS, 1);
    const { overlay } = drawCourt(chartSettings, svgRef);

    handlerMarkerRef.current = overlay
      .append('circle')
      .attr('r', 1)
      .attr('fill', 'red')
      .style('display', 'none')
      .node();

    return () => {
      handlerMarkerRef.current = null;
      if (svgRef.current) {
        svgRef.current.innerHTML = '';
      }
    };
  }, []);

  useEffect(() => {
    const handlerMarker = handlerMarkerRef.current;
    if (!handlerMarker) return;

    if (!handlerXY) {
      handlerMarker.style.display = 'none';
      return;
    }

    handlerMarker.setAttribute('cx', String(handlerXY[0]));
    handlerMarker.setAttribute('cy', String(handlerXY[1]));
    handlerMarker.style.display = 'block';
  }, [handlerXY]);

  return (
    <div className='halfcourt-layout'>
      <div className='halfcourt-surface border border-secondary'>
        <svg ref={svgRef} width='100%' />
      </div>
    </div>
  );
};

export default HalfCourt;
