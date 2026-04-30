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
  const animationFrameRef = useRef<number | null>(null);
  const currentPositionRef = useRef<[number, number] | null>(null);
  const targetPositionRef = useRef<[number, number] | null>(null);

  const stopAnimation = () => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  useEffect(() => {
    if (!svgRef.current) return;

    svgRef.current.innerHTML = '';
    const chartSettings = SHOTCHART_SETTINGS(NBA_SETTINGS, 1);
    const { overlay } = drawCourt(chartSettings, svgRef);

    handlerMarkerRef.current = overlay
      .append('circle')
      .attr('r', 1.7)
      .attr('fill', 'none')
      .attr('stroke', 'green')
      .attr('stroke-width', 0.7)
      .style('display', 'none')
      .node();

    return () => {
      stopAnimation();
      currentPositionRef.current = null;
      targetPositionRef.current = null;
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
      stopAnimation();
      currentPositionRef.current = null;
      targetPositionRef.current = null;
      handlerMarker.style.display = 'none';
      return;
    }

    handlerMarker.style.display = 'block';
    targetPositionRef.current = handlerXY;

    if (!currentPositionRef.current) {
      currentPositionRef.current = [...handlerXY];
      handlerMarker.setAttribute('cx', String(handlerXY[0]));
      handlerMarker.setAttribute('cy', String(handlerXY[1]));
      return;
    }

    if (animationFrameRef.current !== null) {
      return;
    }

    const animate = () => {
      const currentPosition = currentPositionRef.current;
      const targetPosition = targetPositionRef.current;
      const marker = handlerMarkerRef.current;

      if (!currentPosition || !targetPosition || !marker) {
        stopAnimation();
        return;
      }

      const smoothing = 0.1;
      const nextX =
        currentPosition[0] +
        (targetPosition[0] - currentPosition[0]) * smoothing;
      const nextY =
        currentPosition[1] +
        (targetPosition[1] - currentPosition[1]) * smoothing;
      const remainingDistance = Math.hypot(
        targetPosition[0] - nextX,
        targetPosition[1] - nextY,
      );

      currentPositionRef.current = [nextX, nextY];
      marker.setAttribute('cx', String(nextX));
      marker.setAttribute('cy', String(nextY));

      if (remainingDistance < 0.02) {
        currentPositionRef.current = [...targetPosition];
        marker.setAttribute('cx', String(targetPosition[0]));
        marker.setAttribute('cy', String(targetPosition[1]));
        animationFrameRef.current = null;
        return;
      }

      animationFrameRef.current = window.requestAnimationFrame(animate);
    };

    animationFrameRef.current = window.requestAnimationFrame(animate);
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
